#include "ModelResetPlugin.hh"
#include <rclcpp/rclcpp.hpp>
#include <gz/sim/components/JointPosition.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/sim/components/Joint.hh>
#include <gz/sim/components/Link.hh>
#include <gz/sim/components/JointType.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/System.hh>
#include <gz/plugin/Register.hh>
#include <std_msgs/msg/empty.hpp>
#include <gz/math/Rand.hh>
#include <gz/sim/components/JointPositionReset.hh>
#include <gz/sim/components/JointVelocityReset.hh>
#include <gz/sim/components/JointPositionLimitsCmd.hh>
#include <gz/sim/components/JointForceCmd.hh>
#include <gz/sim/components/JointVelocityCmd.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/sim/components/World.hh>
#include <gz/sim/components/AngularVelocityCmd.hh>
#include <gz/sim/components/LinearVelocityCmd.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/Util.hh>

namespace reinforcement_learning
{

// Constructor
ModelResetPlugin::ModelResetPlugin() noexcept
    : node_(nullptr), executor_(nullptr), thread_executor_spin_(), modelEntity(gz::sim::kNullEntity), reset_requested_(false)
{
}

// Destructor
ModelResetPlugin::~ModelResetPlugin() noexcept
{
  // Ensure the executor is properly canceled and the thread is joined
  if (this->executor_) {
    this->executor_->cancel();
  }

  if (this->thread_executor_spin_.joinable()) {
    this->thread_executor_spin_.join();
  }
}

// Configure method to initialize the ROS 2 node and subscribe to reset commands
void ModelResetPlugin::Configure(
    const gz::sim::Entity &entity,
    const std::shared_ptr<const sdf::Element> & /*_sdf*/,
    gz::sim::EntityComponentManager &ecm,
    gz::sim::EventManager &)
{
  // Store the model entity
  this->modelEntity = entity;

  // Ensure ROS 2 is initialized
  if (!rclcpp::ok()) {
    RCLCPP_INFO(rclcpp::get_logger("ModelResetPlugin"), "Initializing ROS 2...");
    rclcpp::init(0, nullptr);
  }

  // Retrieve the model name from the EntityComponentManager (ECM)
  auto modelNameComp = ecm.Component<gz::sim::components::Name>(this->modelEntity);
  std::string modelName = modelNameComp->Data();

  // Create ROS 2 node
  this->node_ = rclcpp::Node::make_shared("model_reset_plugin");

  if (!this->node_) {
    RCLCPP_ERROR(rclcpp::get_logger("ModelResetPlugin"), "Failed to create ROS 2 node!");
  } else {
    RCLCPP_INFO(this->node_->get_logger(), "ROS 2 node created successfully.");
  }

  // Create the ROS 2 MultiThreadedExecutor
  this->executor_ = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
  this->executor_->add_node(this->node_);

  // Spin the executor in a separate thread
  this->thread_executor_spin_ = std::thread([this]() {
    RCLCPP_INFO(this->node_->get_logger(), "Starting to spin the executor...");
    this->executor_->spin();
  });

  // Build the ROS 2 topic path using the model name
  std::string reset_topic = "/" + modelName + "/reset";

  // Set up a subscriber to reset joints on a topic
  this->reset_sub_ = this->node_->create_subscription<std_msgs::msg::Empty>(
    reset_topic, 10,
    [this, &ecm](const std_msgs::msg::Empty::SharedPtr msg) {
      this->ResetJointsCallback(msg, ecm);
    });

  RCLCPP_INFO(this->node_->get_logger(), "ModelResetPlugin is ready with topic: %s", reset_topic.c_str());
}

// Callback to handle the reset request via ROS 2
void ModelResetPlugin::ResetJointsCallback(
    const std_msgs::msg::Empty::SharedPtr /*msg*/,
    gz::sim::EntityComponentManager &ecm)
{
  RCLCPP_INFO(this->node_->get_logger(), "Reset model requested.");
  reset_requested_ = true;
}

// Method to apply joint position and velocity resets
void ModelResetPlugin::ApplyJointPositionReset(gz::sim::EntityComponentManager &ecm)
{
    RCLCPP_INFO(this->node_->get_logger(), "Resetting joint positions");

    auto joints = ecm.EntitiesByComponents(
        gz::sim::components::ParentEntity(this->modelEntity), gz::sim::components::Joint());

    for (auto joint : joints)
    {
        double pos = 0.0;
        ecm.SetComponentData<gz::sim::components::JointPositionReset>(joint, {pos});
        ecm.SetComponentData<gz::sim::components::JointVelocityReset>(joint, {pos});
    }
}

// Method to reset forces and torques acting on the joints
void ModelResetPlugin::ApplyJointForceReset(gz::sim::EntityComponentManager &ecm)
{
    RCLCPP_INFO(this->node_->get_logger(), "Resetting joint forces");

    auto joints = ecm.EntitiesByComponents(
        gz::sim::components::ParentEntity(this->modelEntity), gz::sim::components::Joint());

    for (auto joint : joints)
    {
        // Set joint velocity to 0
        auto velocity = ecm.Component<gz::sim::components::JointVelocityCmd>(joint);
        if (!velocity) {
            ecm.CreateComponent(joint, gz::sim::components::JointVelocityCmd({0.0}));
        } else {
            ecm.SetComponentData<gz::sim::components::JointVelocityCmd>(joint, {0.0});
        }

        // Set joint effort (force/torque) to 0
        auto effort = ecm.Component<gz::sim::components::JointForceCmd>(joint);
        if (!effort) {
            ecm.CreateComponent(joint, gz::sim::components::JointForceCmd({0.0}));
        } else {
            ecm.SetComponentData<gz::sim::components::JointForceCmd>(joint, {0.0});
        }

        // Remove velocity and force commands to clear applied forces
        ecm.RemoveComponent<gz::sim::components::JointVelocityCmd>(joint);
        ecm.RemoveComponent<gz::sim::components::JointForceCmd>(joint);
    }
}

// Method to reset forces and torques acting on the links
void ModelResetPlugin::ApplyLinkForceReset(gz::sim::EntityComponentManager &ecm)
{
    RCLCPP_INFO(this->node_->get_logger(), "Resetting links forces");

    auto links = ecm.EntitiesByComponents(
        gz::sim::components::ParentEntity(this->modelEntity), gz::sim::components::Link());

    for (auto link : links)
    {
      // Set linear velocity to zero
      auto linearVelCmd = ecm.Component<gz::sim::components::LinearVelocityCmd>(link);
      if (!linearVelCmd) {
          ecm.CreateComponent(link, gz::sim::components::LinearVelocityCmd({0, 0, 0}));
      } else {
          ecm.SetComponentData<gz::sim::components::LinearVelocityCmd>(link, {0, 0, 0});
      }

      // Set angular velocity to zero
      auto angularVelCmd = ecm.Component<gz::sim::components::AngularVelocityCmd>(link);
      if (!angularVelCmd) {
          ecm.CreateComponent(link, gz::sim::components::AngularVelocityCmd({0, 0, 0}));
      } else {
          ecm.SetComponentData<gz::sim::components::AngularVelocityCmd>(link, {0, 0, 0});
      }
    }
}

// PreUpdate method to handle joint position reset requests
void ModelResetPlugin::PreUpdate(
    const gz::sim::UpdateInfo &info,
    gz::sim::EntityComponentManager &ecm)
{
    if (reset_requested_)
    {        
        this->ApplyJointPositionReset(ecm);
        reset_requested_ = false;
    }
}

}  

// Register the plugin with Gazebo
GZ_ADD_PLUGIN(reinforcement_learning::ModelResetPlugin, gz::sim::System,
              reinforcement_learning::ModelResetPlugin::ISystemConfigure,
              reinforcement_learning::ModelResetPlugin::ISystemPreUpdate)
#ifndef MODEL_RESET_PLUGIN_HH  // Add include guard
#define MODEL_RESET_PLUGIN_HH

#include <gz/sim/System.hh>  
#include <rclcpp/rclcpp.hpp>  
#include <gz/sim/System.hh>
#include <std_msgs/msg/empty.hpp> 
#include <set>  

namespace reinforcement_learning
{
  class ModelResetPlugin : public gz::sim::System,  
                              public gz::sim::ISystemConfigure,
                              public gz::sim::ISystemPreUpdate
  {
  public:
    ModelResetPlugin() noexcept;
    ~ModelResetPlugin() noexcept;

    // Inherited from gz::sim::ISystemConfigure
    void Configure(const gz::sim::Entity &entity,
                   const std::shared_ptr<const sdf::Element> &sdf,
                   gz::sim::EntityComponentManager &ecm,
                   gz::sim::EventManager &) override;

    // Inherited from gz::sim::ISystemPreUpdate
    void PreUpdate(const gz::sim::UpdateInfo &info,
                   gz::sim::EntityComponentManager &ecm) override;

  private:
    rclcpp::Node::SharedPtr node_;  // ROS 2 node
    std::shared_ptr<rclcpp::executors::MultiThreadedExecutor> executor_;  // ROS 2 executor
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reset_sub_;  // ROS 2 subscriber
    std::thread thread_executor_spin_;  // Thread for spinning the ROS executor
    gz::sim::Entity modelEntity;  // The model entity to which the plugin is attached
    std::atomic<bool> reset_requested_;
    std::mutex ecm_mutex_; 

    // Callback for the reset joints subscriber
    void ResetJointsCallback(const std_msgs::msg::Empty::SharedPtr msg,
                             gz::sim::EntityComponentManager &ecm);

    // Function to apply random joint positions
    void ApplyJointPositionReset(gz::sim::EntityComponentManager &ecm);
    void ApplyJointForceReset(gz::sim::EntityComponentManager &ecm);
    void ApplyLinkForceReset(gz::sim::EntityComponentManager &ecm);

  };
}  

#endif  // MODEL_RESET_PLUGIN_HH
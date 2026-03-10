#ifndef UWBBIAS_H
#define UWBBIAS_H

#include "types/Vec.h"
#include "utils/uvio_sensor_data.h"

namespace uvio {
/**
 * @brief Derived Type class that implements the UWB biases.
 *
 * Contains the constant bias ɣ and the distance depended bias 1 + β. Separated from the anchor class as the biases are dependent on the tag and anchor pair involved in the ranging transaction. 
 */
class UWBBias : public ov_type::Type {
    public: 
        UWBBias(double const_bias_val = 0.0, double dist_bias_val = 0.0) : ov_type::Type(2) {
            Eigen::VectorXd val = Eigen::Vector2d::Zero();
            val << const_bias_val, dist_bias_val;

            _const_bias = std::make_shared<ov_type::Vec>(1);
            _dist_bias = std::make_shared<ov_type::Vec>(1);

            set_value_internal(val);
            set_fej_internal(val);

        }

        inline void update(const Eigen::VectorXd &dx) override {
            assert(dx.rows() == _size);
            set_value(_value + dx);
        }

        inline void set_value(const Eigen::MatrixXd &new_value) override { set_value_internal(new_value); }

        inline void set_fej(const Eigen::MatrixXd &new_value) override {
            set_fej_internal(new_value);
        }

        inline std::shared_ptr<ov_type::Vec> const_bias() { return _const_bias; }

        inline std::shared_ptr<ov_type::Vec> dist_bias() { return _dist_bias; }

        inline std::shared_ptr<ov_type::Type> clone() override {
            // Get values
            double c_val = _const_bias->value()(0);
            double d_val = _dist_bias->value()(0);
            // Make clone
            auto clone = std::make_shared<UWBBias>(c_val, d_val);

            clone->set_value(value());
            clone->set_fej(fej());
            return clone;
        }

    private:

        std::shared_ptr<ov_type::Vec> _const_bias;
        std::shared_ptr<ov_type::Vec> _dist_bias;

        inline void set_value_internal(const Eigen::MatrixXd &new_value){
            _const_bias->set_value(new_value.block(0, 0, 1, 1));
            _dist_bias->set_value(new_value.block(1, 0, 1, 1));
            _value = new_value;
        }

        inline void set_fej_internal(const Eigen::MatrixXd &new_value){
            _const_bias->set_fej(new_value.block(0, 0, 1, 1));
            _dist_bias->set_fej(new_value.block(1, 0, 1, 1));
            _fej = new_value;
        }
};

}

#endif
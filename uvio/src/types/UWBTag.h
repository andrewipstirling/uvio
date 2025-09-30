#pragma once

#include "types/Type.h"
#include "types/Vec.h"
#include <Eigen/Eigen>
#include <memory>

namespace uvio {

struct TagData {
    size_t id;
    bool fix;
    Eigen::Vector3d offset;
};

class UWBTag : public ov_type::Type {
protected:
    size_t _tag_id;                          ///< Unique tag ID
    bool _fixed;                             ///< Fixed or estimated
    float _delay;                            ///< Optional antenna delay
    std::shared_ptr<ov_type::Vec> _p_tz_b;   ///< Position of tag relative to IMU resolved in IMU frame

public:
    UWBTag(const TagData &tag) : ov_type::Type(4), _tag_id(tag.id), _fixed(tag.fix), _delay(0.0) {
        // Allocate Vec for position
        _p_tz_b = std::make_shared<ov_type::Vec>(3);
        _p_tz_b->set_value(tag.offset);

        // Initialize Type's value vector (3 pos + 1 delay)
        Eigen::VectorXd tag_init = Eigen::VectorXd::Zero(4);
        tag_init.segment<3>(0) = tag.offset;
        tag_init(3) = _delay;
        set_value(tag_init);
        set_fej(tag_init);
    }

    ~UWBTag() {}

    /// Update state from perturbation vector
    void update(const Eigen::VectorXd &dx) override {
        assert(dx.rows() == _size);
        set_value(_value + dx);
    }

    /// Clone the tag
    std::shared_ptr<ov_type::Type> clone() override {
        TagData td;
        td.id = _tag_id;
        td.fix = _fixed;
        td.offset = _p_tz_b->value();

        auto clone = std::make_shared<UWBTag>(td);
        clone->set_value(value());
        clone->set_fej(fej());
        return clone;
    }

    /// Accessors
    size_t tag_id() const { return _tag_id; }
    bool fixed() const { return _fixed; }
    float delay() const { return _delay; }
    std::shared_ptr<ov_type::Vec> p_tz_b() const { return _p_tz_b; }
};

} // namespace uvio

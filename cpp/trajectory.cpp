#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

struct Sample {
    double timestamp;
    double x;
    double y;
};

class ConstantVelocityTrajectory {
public:
    ConstantVelocityTrajectory(std::size_t max_history = 20, double max_dt = 0.25)
        : max_history_(max_history), max_dt_(max_dt) {
        if (max_history_ == 0) {
            throw std::invalid_argument("max_history must be positive.");
        }
        if (!std::isfinite(max_dt_) || max_dt_ <= 0.0) {
            throw std::invalid_argument("max_dt must be finite and positive.");
        }
    }

    void add_sample(double timestamp, double x, double y) {
        if (!std::isfinite(timestamp) || !std::isfinite(x) || !std::isfinite(y)) {
            throw std::invalid_argument("timestamps and positions must be finite.");
        }
        if (samples_.size() >= max_history_) {
            samples_.erase(samples_.begin());
        }
        samples_.push_back({timestamp, x, y});
    }

    std::vector<std::pair<double, double>> predict(double horizon, int steps) const {
        if (samples_.size() < 2) {
            throw std::invalid_argument("At least two samples are required for trajectory prediction.");
        }
        if (!std::isfinite(horizon) || horizon < 0.0) {
            throw std::invalid_argument("Horizon must be finite and non-negative.");
        }
        if (steps <= 0) {
            throw std::invalid_argument("Prediction steps must be positive.");
        }

        const auto &last = samples_.back();
        const auto &prev = samples_[samples_.size() - 2];
        double dt = last.timestamp - prev.timestamp;
        if (!std::isfinite(dt) || dt <= 0.0 || dt > max_dt_) {
            throw std::invalid_argument("Invalid or stretched timestamps for constant-velocity model.");
        }

        const std::size_t first = samples_.size() > 5 ? samples_.size() - 5 : 0;
        double mean_t = 0.0;
        double mean_x = 0.0;
        double mean_y = 0.0;
        const std::size_t count = samples_.size() - first;
        for (std::size_t i = first; i < samples_.size(); ++i) {
            mean_t += samples_[i].timestamp;
            mean_x += samples_[i].x;
            mean_y += samples_[i].y;
        }
        mean_t /= static_cast<double>(count);
        mean_x /= static_cast<double>(count);
        mean_y /= static_cast<double>(count);

        double time_spread = 0.0;
        double x_trend = 0.0;
        double y_trend = 0.0;
        for (std::size_t i = first; i < samples_.size(); ++i) {
            const double time_offset = samples_[i].timestamp - mean_t;
            time_spread += time_offset * time_offset;
            x_trend += time_offset * (samples_[i].x - mean_x);
            y_trend += time_offset * (samples_[i].y - mean_y);
        }
        if (time_spread == 0.0) {
            throw std::invalid_argument("Timestamps must not all be equal.");
        }
        const double vx = x_trend / time_spread;
        const double vy = y_trend / time_spread;
        std::vector<std::pair<double, double>> out;
        out.reserve(static_cast<size_t>(steps));
        for (int i = 0; i < steps; ++i) {
            double future_t = horizon * (static_cast<double>(i + 1) / static_cast<double>(steps));
            out.emplace_back(last.x + vx * future_t, last.y + vy * future_t);
        }
        return out;
    }

    size_t size() const { return samples_.size(); }

private:
    std::vector<Sample> samples_;
    std::size_t max_history_;
    double max_dt_;
};

PYBIND11_MODULE(cricket_vision_cpp, m) {
    py::class_<ConstantVelocityTrajectory>(m, "ConstantVelocityTrajectory")
        .def(py::init<std::size_t, double>(), py::arg("max_history") = 20, py::arg("max_dt") = 0.25)
        .def("add_sample", &ConstantVelocityTrajectory::add_sample, py::arg("timestamp"), py::arg("x"), py::arg("y"))
        .def("predict", &ConstantVelocityTrajectory::predict, py::arg("horizon"), py::arg("steps"))
        .def("size", &ConstantVelocityTrajectory::size);
}

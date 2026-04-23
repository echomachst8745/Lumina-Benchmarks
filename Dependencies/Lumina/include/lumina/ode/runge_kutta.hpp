/**
 * @file runge_kutta.hpp
 * 
 * @brief Runge-Kutta ODE solvers.
 * 
 * @ingroup ode
 */
#pragma once

#include <vector>

#include <lumina/ode/euler.hpp>

namespace lumina::ode {

/**
 * @brief Integrates y' = f(t, y) using the 4th order Runge-Kutta method.
 * 
 * @param f ODE right hand side function.
 * @param t0 Start time.
 * @param t_end End time.
 * @param y0 Initial state.
 * @param h Step size.
 * 
 * @return ODEResult with all time points and states.
 * 
 * @ingroup ode
 */
ODEResult RK4(ODEFunction f, double t0, double t_end, State y0, double h);

/**
 * @brief Result of an adaptive step RK45 integration.
 * 
 * @ingroup ode
 */
struct RK45Result
{
    std::vector<double> t;              ///< Accepted time points.
    std::vector<State>  y;              ///< State at each accepted point.
    int                 steps_accepted; ///< Number of accepted steps.
    int                 steps_rejected; ///< Number of rejected steps.
};

/**
 * @brief Integrates y' = f(t, y) using the adaptive Dormand–Prince RK45 method.
 * 
 * @param f ODE right hand side function.
 * @param t0 Start time.
 * @param t_end End time.
 * @param y0 Initial state.
 * @param rtol Relative tolerance.
 * @param atol Absolute tolerance.
 * @param h_init Initial step.
 * 
 * @return RK45Result with accepted time/state pairs and step statistics.
 * 
 * @ingroup ode
 */
RK45Result RK45(ODEFunction f, double t0, double t_end, State y0,
                double r_delta = 1e-6, double a_delta = 1e-9, double h_init = 0.01);

} // namespace lumin::ode
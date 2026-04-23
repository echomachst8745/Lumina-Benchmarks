/**
 * @file euler.hpp
 * 
 * @brief Euler method for solving first order ordinary differential equations.
 * 
 * @ingroup ode
 */
#pragma once

#include <vector>
#include <functional>

namespace lumina::ode {

using State = std::vector<double>; ///< ODE state vector.

/// @brief ODE right hand side function: f(t, y) => dy/dt.
using ODEFunction = std::function<State(double, const State&)>;

/**
 * @brief Result of ODE integration.
 * 
 * @ingroup ode
 */
struct ODEResult
{
    std::vector<double> t; ///< Time points.
    std::vector<State> y; ///< State at each time point.
};

/**
 * @brief Integrates y' = f(t, y) using the forward Euler method.
 * 
 * @param f ODE right hand side function.
 * @param t0 Start time.
 * @param t_end End time.
 * @param y0 Initial state.
 * @param h Step size.
 * 
 * @return ODEResult with all time points and states.
 * 
 * @throws std::invalid_argument if @p h <= 0 or @p t_end <= t0.
 * 
 * @ingroup ode
 */
ODEResult Euler(ODEFunction f, double t0, double t_end, State y0, double h);

/**
 * @brief Integrates y' = f(t, y) using the implicit (backward) Euler method.
 * 
 * @param f ODE right hand side function.
 * @param t0 Start time.
 * @param t_end End time.
 * @param y0 Initial state.
 * @param h Step size.
 * @param max_iter Iterations per step.
 * @param delta Value tolerance.
 * 
 * @return ODEResult with all time points and states.
 * 
 * @throws std::invalid_argument if @p h <= 0 or @p t_end <= t0.
 * 
 * @ingroup ode
 */
ODEResult EulerImplicit(ODEFunction f, double t0, double t_end, State y0, double h,
                        int max_iter = 100, double delta = 1e-10);

} // namespace lumina::ode
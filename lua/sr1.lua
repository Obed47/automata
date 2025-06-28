-- SR1 Quasi-Newton Method in Lua
local function sr1_quasi_newton(f, grad_f, x0, max_iter, tol)
    -- Parameters
    local n = #x0
    local H = {} -- Approximate inverse Hessian (initialized as identity matrix)
    for i = 1, n do
        H[i] = {}
        for j = 1, n do
            H[i][j] = (i == j) and 1 or 0
        end
    end

    local x = {}
    for i = 1, n do x[i] = x0[i] end

    local g = grad_f(x)
    local iter = 0

    while iter < max_iter do
        -- Check convergence
        local grad_norm = 0
        for i = 1, n do grad_norm = grad_norm + g[i]^2 end
        grad_norm = math.sqrt(grad_norm)
        if grad_norm < tol then break end

        -- Compute search direction: p = -H * g
        local p = {}
        for i = 1, n do
            p[i] = 0
            for j = 1, n do
                p[i] = p[i] - H[i][j] * g[j]
            end
        end

        -- Backtracking line search (Armijo condition)
        local alpha = 1.0
        local c = 1e-4
        local rho = 0.5
        local fx = f(x)
        local new_x = {}
        for i = 1, n do new_x[i] = x[i] + alpha * p[i] end

        while f(new_x) > fx + c * alpha * grad_norm^2 do
            alpha = alpha * rho
            for i = 1, n do new_x[i] = x[i] + alpha * p[i] end
        end

        -- Update x and gradient
        local new_g = grad_f(new_x)
        local s, y = {}, {}
        for i = 1, n do
            s[i] = new_x[i] - x[i]
            y[i] = new_g[i] - g[i]
        end

        -- SR1 Update: H = H + (s - Hy)(s - Hy)^T / (s - Hy)^T y
        local Hy = {}
        for i = 1, n do
            Hy[i] = 0
            for j = 1, n do
                Hy[i] = Hy[i] + H[i][j] * y[j]
            end
        end

        local s_minus_Hy = {}
        for i = 1, n do
            s_minus_Hy[i] = s[i] - Hy[i]
        end

        local denominator = 0
        for i = 1, n do
            denominator = denominator + s_minus_Hy[i] * y[i]
        end

        if math.abs(denominator) > tol then
            for i = 1, n do
                for j = 1, n do
                    H[i][j] = H[i][j] + (s_minus_Hy[i] * s_minus_Hy[j]) / denominator
                end
            end
        end

        -- Update for next iteration
        x = new_x
        g = new_g
        iter = iter + 1
    end

    return x, iter
end

--- Example: Minimize f(x) = x1^2 + x2^2 + x1*x2
local function example_f(x)
    return x[1]^2 + x[2]^2 + x[1]*x[2]
end

local function example_grad_f(x)
    return {2*x[1] + x[2], 2*x[2] + x[1]}
end

-- Run SR1 on the example
local x0 = {4, -3}  -- Initial guess
local solution, iterations = sr1_quasi_newton(example_f, example_grad_f, x0, 100, 1e-6)

print(string.format("Solution: x1 = %.6f, x2 = %.6f", solution[1], solution[2]))
print("Iterations:", iterations)
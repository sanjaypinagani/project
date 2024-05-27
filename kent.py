def policy_iteration(P, R, states, actions, gamma=0.9):
    policy = np.zeros(len(states), dtype=int)  # Initialize policy to the first action
    while True:
        V = policy_evaluation(policy, P, R, states, actions, gamma)
        policy_stable = True
        for s in range(len(states)):
            old_action = policy[s]
            policy[s] = np.argmax([sum(P[a][s, s_prime] * (R[s, ai] + gamma * V[s_prime])
                                       for s_prime in range(len(states)))
                                   for ai, a in enumerate(actions)])
            if old_action != policy[s]:
                policy_stable = False
        if policy_stable:
            break
    return policy, V

# Run Value Iteration
V = value_iteration(P, R, states, actions)
print("Optimal Value Function (Value Iteration):", V)

# Run Policy Iteration
policy, V = policy_iteration(P, R, states, actions)
print("Optimal Policy (Policy Iteration):", policy)
print("Optimal Value Function (Policy Iteration):",V)

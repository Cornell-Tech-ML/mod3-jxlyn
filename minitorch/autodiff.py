from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol, Set


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    if not (0 <= arg < len(vals)):
        raise ValueError(
            f"Invalid arg index {arg}, must be between 0 and {len(vals) - 1}"
        )

    vals_positive = list(vals)
    vals_negative = list(vals)

    vals_positive[arg] += epsilon
    vals_negative[arg] -= epsilon

    f_positive = f(*vals_positive)
    f_negative = f(*vals_negative)

    derivative = (f_positive - f_negative) / (2 * epsilon)
    return derivative


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate a derivative value in the variable.

        Args:
        ----
            x : The value to accumulate.

        Returns:
        -------
            None

        """
        ...

    @property
    def unique_id(self) -> int:
        """A unique identifier for the variable.

        Returns
        -------
            An integer identifier.

        """
        ...

    def is_leaf(self) -> bool:
        """Check if the variable is a leaf node.

        Returns
        -------
            True if the variable is a leaf node.

        """
        ...

    def is_constant(self) -> bool:
        """Check if the variable is a constant node.

        Returns
        -------
            True if the variable is a constant node.

        """
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Get the parents of the variable.

        Returns
        -------
            An iterable of the parents of the variable.

        """
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Compute the chain rule for the variable.

        Args:
        ----
            d_output : The value of the derivative of the output.

        Returns:
        -------
            An iterable of pairs of the parent and the derivative value.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    visited: Set[Variable] = set()
    topo_order: List[Variable] = []

    def dfs(var: Variable) -> None:
        if var not in visited:
            visited.add(var)
            # Only traverse the parents if the variable is not a constant
            if not var.is_constant():
                for parent in var.parents:
                    dfs(parent)
                topo_order.append(var)

    dfs(variable)
    return reversed(topo_order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    # TODO: Implement for Task 1.4.
    topo_order = topological_sort(variable)
    var_to_derivative = {variable: deriv}

    # Traverse the variables in topological order
    for var in topo_order:
        d_output = var_to_derivative.get(var, 0)  # Get the current derivative

        # If the variable is a leaf, accumulate the derivative
        if var.is_leaf():
            var.accumulate_derivative(d_output)
        else:
            # Otherwise, apply the chain rule to propagate derivatives to parents
            for parent, gradient in var.chain_rule(d_output):
                if parent in var_to_derivative:
                    var_to_derivative[parent] += gradient  # Accumulate the derivative
                else:
                    var_to_derivative[parent] = gradient  # Initialize derivative


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved values.

        Args:
        ----
            None

        Returns:
        -------
            The saved values.

        """
        return self.saved_values

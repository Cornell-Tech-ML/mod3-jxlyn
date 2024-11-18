from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the function to a set of Scalar values.

        Args:
        ----
            *vals : The input values to the function.

        Returns:
        -------
            Scalar : The output value.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for addition

        Args:
        ----
            ctx : Context object
            a : float
            b : float

        Returns:
        -------
            float : a + b

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for addition

        Args:
        ----
            ctx : Context object
            d_output : float

        Returns:
        -------
            Tuple[float, float] : d_output, d_output

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for log

        Args:
        ----
            ctx : Context object
            a : float

        Returns:
        -------
            float : log(a)

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for log

        Args:
        ----
            ctx : Context object
            d_output : float

        Returns:
        -------
            float : d_output / a

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for multiplication

        Args:
        ----
            ctx : Context object
            a : float
            b : float

        Returns:
        -------
            float : a * b

        """
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for multiplication

        Args:
        ----
            ctx : Context object
            d_output : float

        Returns:
        -------
            Tuple[float, float] : b * d_output, a * d_output

        """
        a, b = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for inverse

        Args:
        ----
            ctx : Context object
            a : float

        Returns:
        -------
            float : 1 / a

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for inverse

        Args:
        ----
            ctx : Context object
            d_output : float

        Returns:
        -------
            float : -d_output / a^2

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for negation

        Args:
        ----
            ctx : Context object
            a : float

        Returns:
        -------
            float : -a

        """
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for negation

        Args:
        ----
            ctx : Context object
            d_output : float

        Returns:
        -------
            float : -d_output

        """
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = (1 / (1 + exp(-x)))$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for sigmoid

        Args:
        ----
            ctx : Context object
            a : float

        Returns:
        -------
            float : 1 / (1 + exp(-a))

        """
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for sigmoid

        Args:
        ----
            ctx : Context object
            d_output : float

        Returns:
        -------
            float : sigmoid(a) * (1 - sigmoid(a)) * d_output

        """
        (a,) = ctx.saved_values
        return operators.sigmoid(a) * (1 - operators.sigmoid(a)) * d_output


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for ReLU

        Args:
        ----
            ctx : Context object
            a : float

        Returns:
        -------
            float : max(0, a)

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for ReLU

        Args:
        ----
            ctx : Context object
            d_output : float

        Returns:
        -------
            float : (a > 0) * d_output

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function $f(x) = exp(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for exponential

        Args:
        ----
            ctx : Context object
            a : float

        Returns:
        -------
            float : exp(a)

        """
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for exponential

        Args:
        ----
            ctx : Context object
            d_output : float

        Returns:
        -------
            float : exp(a) * d_output

        """
        (a,) = ctx.saved_values
        return operators.exp(a) * d_output


class LT(ScalarFunction):
    """Less-than function $f(x, y) = 1.0$ if $x < y$ else $0.0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for less-than

        Args:
        ----
            ctx : Context object
            a : float
            b : float

        Returns:
        -------
            float : 1.0 if a < b else 0.0

        """
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for less-than

        Args:
        ----
            ctx : Context object
            d_output : float

        Returns:
        -------
            Tuple[float, float] : 0.0, 0.0

        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equality function $f(x, y) = 1.0$ if $x == y$ else $0.0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for equality

        Args:
        ----
            ctx : Context object
            a : float
            b : float

        Returns:
        -------
            float : 1.0 if a == b else 0.0

        """
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for equality

        Args:
        ----
            ctx : Context object
            d_output : float

        Returns:
        -------
            Tuple[float, float] : 0.0, 0.0

        """
        return 0.0, 0.0

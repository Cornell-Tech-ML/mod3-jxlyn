"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:


def mul(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The product of x and y.

    """
    return x * y


def id(x: float) -> float:
    """Identity function.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The input value.

    """
    return x


def add(x: float, y: float) -> float:
    """Add two floats.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The sum of x and y.

    """
    return x + y


def neg(x: float) -> float:
    """Negate a float.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The negation of x.

    """
    return -x


def lt(x: float, y: float) -> float:
    """Less than comparison.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: 1.0 if x is less than y, 0.0 otherwise.

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Equality comparison.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        1.0 if x is equal to y, 0.0 otherwise.

    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Maximum of two floats.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The maximum of x and y.

    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Equality comparison with tolerance.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: 1.0 if the absolute difference between x and y is less than 1e-2, 0.0 otherwise.

    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Sigmoid function.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The sigmoid of x.

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Relu function

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The relu of x.

    """
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Logarithm function.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The natural logarithm of x.

    """
    return math.log(x)


def exp(x: float) -> float:
    """Exponential function.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The exponential of x.

    """
    return math.exp(x)


def log_back(x: float, b: float) -> float:
    """Logarithm function backpropagation.

    Args:
    ----
        x (float): The input value.
        b (float): The gradient.

    Returns:
    -------
        float: The gradient of the logarithm of x.

    """
    return b / x


def inv(x: float) -> float:
    """Inverse function.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The inverse of x.

    """
    return 1.0 / x


def inv_back(x: float, b: float) -> float:
    """Inverse function backpropagation.

    Args:
    ----
        x (float): The input value.
        b (float): The gradient.

    Returns:
    -------
        float: The gradient of the inverse of x.

    """
    return -b / (x * x)


def relu_back(x: float, b: float) -> float:
    """Relu function backpropagation.

    Args:
    ----
        x (float): The input value.
        b (float): The gradient.

    Returns:
    -------
        float: The gradient of the relu of x.

    """
    return b if x > 0 else 0.0


#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions


def map(fn: Callable[[float], float], xs: Iterable[float]) -> Iterable[float]:
    """Map a function over a list.

    Args:
    ----
        fn (Callable[[float], float]): The function to apply.
        xs (Iterable[float]): The list of floats.

    Returns:
    -------
        Iterable[float]: The list of floats after applying the function.

    """
    return [fn(x) for x in xs]


def zipWith(
    fn: Callable[[float, float], float], xs: Iterable[float], ys: Iterable[float]
) -> Iterable[float]:
    """Map a function over two lists.

    Args:
    ----
        fn (Callable[[float, float], float]): The function to apply.
        xs (Iterable[float]): The first list of floats.
        ys (Iterable[float]): The second list of floats.

    Returns:
    -------
        Iterable[float]: The list of floats after applying the function.

    """
    return [fn(x, y) for x, y in zip(xs, ys)]


def reduce(
    fn: Callable[[float, float], float], xs: Iterable[float], init: float
) -> float:
    """Reduce a list under a function.

    Args:
    ----
        fn (Callable[[float, float], float]): The function to apply.
        xs (Iterable[float]): The list of floats.
        init (float): The initial value.

    Returns:
    -------
        float: The reduced value.

    """
    acc = init
    for x in xs:
        acc = fn(acc, x)
    return acc


def negList(xs: Iterable[float]) -> Iterable[float]:
    """Negate a list of floats.

    Args:
    ----
        xs (Iterable[float]): The list of floats.

    Returns:
    -------
        Iterable[float]: The list of negated floats.

    """
    return map(neg, xs)


def addLists(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
    """Add two lists elementwise.

    Args:
    ----
        xs (Iterable[float]): The first list of floats.
        ys (Iterable[float]): The second list of floats.

    Returns:
    -------
        Iterable[float]: The list of floats after adding the two lists elementwise.

    """
    return zipWith(add, xs, ys)


def sum(xs: Iterable[float]) -> float:
    """Sum a list of floats.

    Args:
    ----
        xs (Iterable[float]): The list of floats.

    Returns:
    -------
        float: The sum of the list of floats.

    """
    return reduce(add, xs, 0.0)


def prod(xs: Iterable[float]) -> float:
    """Product of a list of floats.

    Args:
    ----
        xs (Iterable[float]): The list of floats.

    Returns:
    -------
        float: The product of the list of floats.

    """
    return reduce(mul, xs, 1.0)

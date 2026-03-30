# CS310 Natural Language Processing

## Assignment 2 (written) Answers

## 1) Show that the cross-entropy loss is J(y, y_hat) = -log(y_hat_o)

Since y is a one-hot vector for the true outside word o, we have y_o = 1 and y_w = 0 for every w != o.

```text
J(y, y_hat) = -sum_w y_w log(y_hat_w)
            = -(y_o log(y_hat_o) + sum_{w != o} y_w log(y_hat_w))
            = -(1 * log(y_hat_o) + sum_{w != o} 0 * log(y_hat_w))
            = -log(y_hat_o)
```

In words, cross-entropy with a one-hot target keeps only the log-probability assigned to the correct outside word.

## 2) Compute partial J / partial v_c

Let

```text
z = U^T v_c,    y_hat = softmax(z),    J(y, y_hat) = -sum_w y_w log(y_hat_w)
```

For softmax followed by cross-entropy, the derivative with respect to the score vector is

```text
partial J / partial z = y_hat - y
```

Because z = U^T v_c, applying the chain rule gives

```text
partial J / partial v_c
= U (partial J / partial z)
= U (y_hat - y)
```

So the final answer is

```text
partial J / partial v_c = U (y_hat - y)
```

Equivalently, in expanded form,

```text
partial J / partial v_c = sum_w (y_hat_w - y_w) u_w
```

## 3) Compute partial J / partial u_w for each outside word vector

For each outside word vector u_w, the corresponding score is

```text
z_w = u_w^T v_c
```

Using the chain rule,

```text
partial J / partial u_w
= (partial J / partial z_w) (partial z_w / partial u_w)
= (y_hat_w - y_w) v_c
```

Therefore there are two cases.

For the true outside word w = o,

```text
partial J / partial u_o = (y_hat_o - 1) v_c
```

For every other outside word w != o,

```text
partial J / partial u_w = y_hat_w v_c
```

A compact equivalent form is

```text
partial J / partial u_w = (y_hat_w - y_w) v_c
```

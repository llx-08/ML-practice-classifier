# Logistic Regression

### Sigmoid Func in Logistic Regression

$sigmoid:$
$$
\sigma(x) = \frac 1 {1+e^{-x}}
$$


在每个特征上乘上回归系数，再将结果代入$Sigmoid$函数中，把数值范围缩小到$(0, 1)$，大于0.5的数据被分为1类，否则分为0类。
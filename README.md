
# F16 Benchmark Data Analysis

The F16 Benchmark is a GVT on an F16 aircraft carrier. Its primary area of interest is surrounding the nonlinear connection between the wing and the payload. This analysis aims to identify the nonlinearities, quantifying them where possible. Ultimately, we want to build a model that describes the system dynamics using the given training data.

See 'Nonlinear ground vibration identification of an F-16 aircraft Part I – Fast nonparametric analysis of distortions in FRF measurements' for more details about how the GVT was conducted.

See https://www.nonlinearbenchmark.org/benchmarks/f-16-gvt for the training data.

# Restoring Force ODE

The restoring force method involves building a restoring force model that captures the nonlinearities at the connection. We can then solve an ODE to obtain our states.



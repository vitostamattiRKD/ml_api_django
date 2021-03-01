### ML API with Django

#### Models
Endpoint - to keep information about our endpoints,
MLAlgorithm - to keep information about ML algorithms used in the service,
MLAlgorithmStatus - to keep information about ML algorithm statuses. The status can change in time, for example, we can set testing as initial status and then after testing period switch to production state.
MLRequest - to keep information about all requests to ML algorithms. It will be needed to monitor ML algorithms and run A/B tests.
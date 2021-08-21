Deploy with `Docker`
```shell
docker build . --tag paper-finder
docker run --network=host paper-finder
```



## Updates

### August 21 2021
Optimize cosine similartiy calculation.
This reduces similarity calculation for a typical 10 token user input by 50× _(from ~50ms -> 1ms)_
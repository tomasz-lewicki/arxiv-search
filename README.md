Deploy to `Heroku`

```shell
git push https://git.heroku.com/guarded-woodland-26275.git
```

Deploy with `Docker`
```
docker run --network=host arxiv-app
```


```shell
curl -d '{"abstract":"Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers---8x deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers."}' -H "Content-Type: application/json" -X POST http://localhost:5000/recommend
```


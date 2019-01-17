# Convert my model

### Convert my model network file

``` python
network = my_model.to_json() # call keras model to_json function and save model architecture
network_file = open(args.network_path, 'w')
network_file.write(network)
network_file.close() 
```

### Convert my model weights file 

``` python
from convert_weights import *
weights = convert_weights(my_model)
weights = array(weights, 'float32')
weights_file = open(args.weights_path, 'wb')
weights.tofile(weights_file)
weights_file.close()  
```

Your model's layers must contain in our [support layer list](https://github.com/luozhiping/neural_network_on_ios/blob/master/Document/layer_list.markdown)
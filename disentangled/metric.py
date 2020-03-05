import numpy as np
import tensorflow as tf
import disentangled.dataset as dataset import disentangled.model.utils
import disentangled.utils as utils


_FACTORS_IN_ORDER = [
    "label_floor_hue",
    "label_wall_hue",
    "label_object_hue",
    "label_scale",
    "label_shape",
    "label_orientation",
]
_NUM_VALUES_PER_FACTOR = {
    "label_floor_hue": 10,
    "label_wall_hue": 10,
    "label_object_hue": 10,
    "label_scale": 8,
    "label_shape": 4,
    "label_orientation": 15,
}


def label_map(d):
    labels = tf.convert_to_tensor([d[f] for f in _FACTORS_IN_ORDER], dtype=tf.uint8)

    return {"image": d["image"] / 255, "label": labels}


def fix_factor(factor, factor_value):
    return lambda x: x["label"][factor] == factor_value

def majority_voting_classifier(data, n_latent, n_generative):
    V = np.zeros((n_latent, n_generative))

    for d, k in data:
        V[d,k] += 1
   
        return V

if __name__ == "__main__":
    utils.disable_info_output()

    model = disentangled.model.utils.load("betavae_shapes3d")

    full_data = dataset.shapes3d.pipeline()

    std_dev = []
    for batch in full_data.take(10):
        std_dev.append(tf.math.reduce_std(model.encode(batch)[0], axis=0))
    
    empirical_std_dev = tf.reduce_mean(tf.stack(std_dev, axis=0), keepdims=True, axis=0)
    
    # PRUNE COLLAPSED LATENTS

    labeled_data = (
        dataset.shapes3d.load()
        .map(label_map)
    )
      
    samples = []

    for i in range(1300):
        factor = np.random.randint(0, len(_FACTORS_IN_ORDER))
        factor_value = np.random.randint(0, _NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[factor]])

        labeled_data.filter(fix_factor(factor, factor_value)).map(dataset.utils.get_image).batch(64).take(1)

        representations = model.encode(batch)[0] 
        representations /= empirical_std_dev
        representations_variance = tf.math.reduce_variance(representations, axis=0)
        dimension = tf.argmin(representations_variance)
    
        samples.append(tf.stack((dimension, factor)))
    

    samples = tf.stack(samples)

    train_data, test_data = np.split(samples, [500])

    V = majority_voting_classifier(train_data, 32, 6)
    classifier = np.argmax(V, axis=1)
       
    test_estimate = classifier[test_data[:,0]]
    target = test_data[:,1]

    error_rate = np.count_nonzero(target - test_estimate) / target.size
    print("Error Rate: {:%}".format(error_rate))

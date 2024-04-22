
import * as tf from '@tensorflow/tfjs';

import { SymbolicTensor } from '@tensorflow/tfjs';


export const IMAGE_SIZE = 224;


/**
 * the metadata to describe the model's creation,
 * includes the labels associated with the classes
 * and versioning information from training.
 */
export interface Metadata {
    tfjsVersion: string;
    tmVersion?: string;
    packageVersion: string;
    packageName: string;
    modelName?: string;
    timeStamp?: string;
    labels: string[];
    userMetadata?: {};
    grayscale?: boolean;
    imageSize?: number;
}

export interface ModelOptions {
    version?: number;
    checkpointUrl?: string;
    alpha?: number;
    trainingLayer?: string;
}

const DEFAULT_MOBILENET_VERSION = 2;
const DEFAULT_TRAINING_LAYER_V1 = 'conv_pw_13_relu';
const DEFAULT_TRAINING_LAYER_V2 = "out_relu"; 
const DEFAULT_ALPHA_V1_v2 = 0.35;
const DEFAULT_ALPHA_V1 = 0.25;//256
const DEFAULT_ALPHA_V2 = 0.5; //512
const DEFAULT_ALPHA_V3 = 0.75;//768 features
const DEFAULT_ALPHA_V4 = 1;//1024 features
const DEFAULT_ALPHA = 1;//1024 features

// v2: 0.35, 0.50, 0.75 or 1.00.

const isAlphaValid = (version: number, alpha: number) => {
    if (version === 1) {
        if (alpha !== 0.25 && alpha !== 0.5 && alpha !== 0.75 && alpha !== 1) {
            console.warn("Invalid alpha. Options are: 0.25, 0.50, 0.75 or 1.00.");
            console.log("Loading model with alpha: ", DEFAULT_ALPHA_V1.toFixed(2)); 
            return DEFAULT_ALPHA_V1;
        }
    }
    else {
        if (alpha !== 0.35 && alpha !== 0.5 && alpha !== 0.75 && alpha !== 1) {
            console.warn("Invalid alpha. Options are: 0.35, 0.50, 0.75 or 1.00.");
            console.log("Loading model with alpha: ", DEFAULT_ALPHA_V2.toFixed(2)); 
            return DEFAULT_ALPHA_V2;
        }
    }

    return alpha;
};


const parseModelOptions = (options?: ModelOptions) => {
    options = options || {}

    if (options.checkpointUrl && options.trainingLayer) {
        if (options.alpha || options.version){
            console.warn("Checkpoint URL passed to modelOptions, alpha options are ignored");
        }        
        return [options.checkpointUrl, options.trainingLayer];
    } else {
        options.version = options.version || DEFAULT_MOBILENET_VERSION;
        
        if(options.version === 1){
            options.alpha = options.alpha || DEFAULT_ALPHA_V4;  
            options.alpha = isAlphaValid(options.version, options.alpha);

            console.log(`Loading mobilenet ${options.version} and alpha ${options.alpha}`);
            // exception is alpha o f 1 can only be 1.0
            let alphaString = options.alpha.toFixed(2);
            if (alphaString === "1.00") { alphaString = "1.0"; }

            console.log("Using the model: ",  )

            return [
                // tslint:disable-next-line:max-line-length
                //They are loading MobileNet_v1        
                `https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_${alphaString}_${IMAGE_SIZE}/model.json`,
                DEFAULT_TRAINING_LAYER_V1
            ];
        }
        else if (options.version === 2){
            options.alpha = options.alpha || DEFAULT_ALPHA_V4;  
            options.alpha = isAlphaValid(options.version, options.alpha);

            console.log(`Loading mobilenet ${options.version} and alpha ${options.alpha}`);
            console.log(`Loading mobilenet ${options.version} and alpha ${options.alpha}`);

            return [
                // tslint:disable-next-line:max-line-length        
                `https://storage.googleapis.com/teachable-machine-models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_${options.alpha.toFixed(2)}_${IMAGE_SIZE}_no_top/model.json`,
                DEFAULT_TRAINING_LAYER_V2
            ];
        } else {
            throw new Error(`MobileNet V${options.version} doesn't exist`);
        }   
    }
};


/**
 * load the base mobilenet model
 * @param modelOptions options determining what model to load
 */
export async function loadTruncatedMobileNet(modelOptions?: ModelOptions) {
    const [checkpointUrl, trainingLayer] = parseModelOptions(modelOptions);
    
    
    const mobilenet = await tf.loadLayersModel(checkpointUrl);

    if (modelOptions && modelOptions.version === 1){
        const layer = mobilenet.getLayer(trainingLayer);
        
        const truncatedModel = tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
        console.log("Feature model loaded, memory: ", tf.memory().numBytes);

        
        const model = tf.sequential();
        model.add(truncatedModel);
        model.add(tf.layers.flatten());        
        
        
        return model;
    }
    else {
        const layer = mobilenet.getLayer(trainingLayer);
        const truncatedModel = tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
        console.log("Feature model loaded, memory: ", tf.memory().numBytes);
        const model = tf.sequential();
        model.add(truncatedModel);
        model.add(tf.layers.globalAveragePooling2d({})); // go from shape [7, 7, 1280] to [1280]
        return model;
    }
}


export class CustomMobileNet {
    /**
     * the truncated mobilenet model we will train on top of
     */
    protected truncatedModel!: tf.LayersModel;
    static truncatedModel: tf.Sequential;

static getinputShape(){
   /**truncatedModel is the base model, the model used to apply transfer learning */
  const inputShape: any = this.truncatedModel.outputs[0].shape.slice(1); // [ 7 x 7 x 1280] (not sure about those dimensions)
//   console.log("Input Shape(complete): ", this.truncatedModel.outputs[0].shape);
//   console.log("Input Shape: ", inputShape);

  const inputSize = tf.util.sizeFromShape(inputShape);

//   console.log("Input Size: ", inputSize);

  return inputSize;
}

    static get EXPECTED_IMAGE_SIZE() {
        return IMAGE_SIZE;
    }

    protected _metadata!: Metadata;

    public getMetadata() {
        return this._metadata;
    }

    constructor() {
        // this._metadata = fillMetadata(metadata);
        //Loading the truncated model
        // loadTruncatedMobileNet();
        // this.loadFeatureModel();

    }

   static  async loadFeatureModel(){

        this.truncatedModel = await loadTruncatedMobileNet();

    }

    /**
     * get the total number of classes existing within model
     */
    // getTotalClasses() {
    //     const output = this.model.output as SymbolicTensor;
    //     const totalClasses = output.shape[1];
    //     return totalClasses;
    // }

    /**
     * get the model labels
     */
    getClassLabels() {
        return this._metadata.labels;
    }

    /**
     * Given an image element, makes a prediction through mobilenet returning the
     * probabilities of the top K classes.
     * @param image the image to classify
     * @param maxPredictions the maximum number of classification predictions
     */
    // async predictTopK(image: ClassifierInputSource, maxPredictions = 10, flipped = false) {
    //     const croppedImage = cropTo(image, this._metadata.imageSize, flipped);

    //     const logits = tf.tidy(() => {
    //         const captured = capture(croppedImage, this._metadata.grayscale);
    //         return this.model.predict(captured);
    //     });

    //     // Convert logits to probabilities and class names.
    //     const classes = await getTopKClasses(this._metadata.labels, logits as tf.Tensor<tf.Rank>, maxPredictions);
    //     dispose(logits);

    //     return classes;
    // }

    /**
     * Given an image element, makes a prediction through mobilenet returning the
     * probabilities for ALL classes.
     * @param image the image to classify
     * @param flipped whether to flip the image on X
     */
    // async predict(image: ClassifierInputSource, flipped = false) {
    //     const croppedImage = cropTo(image, this._metadata.imageSize, flipped);

    //     const logits = tf.tidy(() => {
    //         const captured = capture(croppedImage, this._metadata.grayscale);
    //         return this.model.predict(captured);
    //     });

    //     const values = await (logits as tf.Tensor<tf.Rank>).data();

    //     const classes = [];
    //     for (let i = 0; i < values.length; i++) {
    //         classes.push({
    //             className: this._metadata.labels[i],
    //             probability: values[i]
    //         });
    //     }

    //     dispose(logits);

    //     return classes;
    // }

    public dispose() {
        this.truncatedModel.dispose();
    }
}// end of CustomMobileNet

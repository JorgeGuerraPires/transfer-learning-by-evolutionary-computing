import { CustomMobileNet, loadTruncatedMobileNet } from "./custom-mobilenet";

import * as tf from '@tensorflow/tfjs';

import { capture } from '../utils/tf';
import { Class } from '../utils/class';
import { Util } from "../utils/util";

import * as seedrandom from 'seedrandom';

import * as tfvis from '@tensorflow/tfjs-vis';

import { TensorContainer } from '@tensorflow/tfjs-core/dist/tensor_types';

const VALIDATION_FRACTION = 0.15;



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

/**
 * Receives a Metadata object and fills in the optional fields such as timeStamp
 * @param data a Metadata object
 */
const fillMetadata = (data: Partial<Metadata>) => {
  // util.assert(typeof data.tfjsVersion === 'string', () => `metadata.tfjsVersion is invalid`);
  // data.packageVersion = data.packageVersion || version;
  data.packageName = data.packageName || '@teachablemachine/image';
  data.timeStamp = data.timeStamp || new Date().toISOString();
  data.userMetadata = data.userMetadata || {};
  data.modelName = data.modelName || 'untitled';
  data.labels = data.labels || [];
  // data.imageSize = data.imageSize || IMAGE_SIZE;
  return data as Metadata;
};

export class TeachableMobileNet extends CustomMobileNet {

    // Array of all the examples collected.
    /**
      It is static since all the instance will share the same features, for saving memory and time.
      The idea is avoiding restoring the features individually and having the recalculate them for every new
      individuals.
     */
    public static examples: Float32Array[][] = [];

    // Number of total samples
    private static totalSamples = 0;

    classes: Class[]=[];

    static classes_names: string[]=[];

    static numClasses: number;

        /**
     * the training model for transfer learning
     */
    protected trainingModel!: tf.LayersModel;
  trainDataset: any;
  validationDataset: any;
  static trainDataset: any;
  static validationDataset: any;
  static numValidation: number;

  constructor(){
    super();
    this.createHead();
  }

  /**
   * This method will return the head, the trainable part, the part under evolution.
   */
  getHead(){
    return this.trainingModel;
  }

 /**
  * Create the head for transfer learning.
  * This is the trainable section of the transfer learning.
  */
 createHead(){

  const inputSize= TeachableMobileNet.getinputShape();

  this.trainingModel = tf.sequential({
    layers: [
      tf.layers.dense({
        inputShape: [inputSize],
        units:100,
        activation: 'relu',  
        useBias: true
    }),
    tf.layers.dense({  
      useBias: false,
      activation: 'softmax',
      units: TeachableMobileNet.classes_names.length
  })
    ]
  });

  const optimizer = tf.train.adam();
  // const optimizer = tf.train.rmsprop(params.learningRate);

  this.trainingModel.compile({
      optimizer,
      // loss: 'binaryCrossentropy',
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
  });
 }
 
 async  train(){

  const trainingSurface = { name: 'Loss and MSE', tab: 'Training' };

  const dataset= TeachableMobileNet.convertToTfDataset();

  //Salving a copy of the validation dataset, for later
  TeachableMobileNet.validationDataset= dataset.validationDataset;

  // console.log("Dataset for training: ", dataset.trainDataset);

  const trainData = dataset.trainDataset.batch(30);
  const validationData = dataset.validationDataset.batch(10);

  // this.createHead();

  const callbacks= [
    // Show on a tfjs-vis visor the loss and accuracy values at the end of each epoch.
    tfvis.show.fitCallbacks(trainingSurface, ['loss', 'acc', "val_loss", "val_acc"],
      {
        callbacks: ['onEpochEnd'],
      }
    ),
    {},]

  const history: any = await this.trainingModel.fitDataset(trainData, {
    epochs: 100,
    validationData,
    callbacks
}).then((info:any)=>{
   console.log('Precisão final', info.history.val_acc[info.history.acc.length-1]);
});


// await this.accuracy_per_class();


// console.log("History: ", history.history.acc);
  // await this.trainingModel.fit(this.featureX, this.target, {})
 }

async accuracy_per_class(confusion_matrix_recipient: any){
  /**Calculating Accuracy per class */
const accuracyperclass: any =  await this.calculateAccuracyPerClass(TeachableMobileNet.validationDataset);
// console.log("Accuracy per class: ", accuracyperclass);

//Confusion matrix
// Calling tf.confusionMatrix() method 
const output = tf.math.confusionMatrix( accuracyperclass.reference, accuracyperclass.predictions, TeachableMobileNet.classes_names.length); 
  
// Printing output 
output.print()
const confusion_matrix=  output.dataSync();

// console.log(confusion_matrix);
// console.log(confusion_matrix[TeachableMobileNet.classes_names.length + TeachableMobileNet.classes_names.length]);

const accuracy = [];

for(let i=0; i<TeachableMobileNet.classes_names.length;i++){

  accuracy.push(confusion_matrix[TeachableMobileNet.classes_names.length*i+ i]/TeachableMobileNet.numValidation)
}

console.log("Accuracy per class: ", accuracy);

for(let i=0; i<TeachableMobileNet.classes_names.length;i++){

  confusion_matrix_recipient.push([]);

  for(let j=0; j<TeachableMobileNet.classes_names.length;j++){
    confusion_matrix_recipient[i].push([]);

    confusion_matrix_recipient[i][j]= confusion_matrix[TeachableMobileNet.classes_names.length*i+ j]/TeachableMobileNet.numValidation
    confusion_matrix_recipient[i][j]=(confusion_matrix_recipient[i][j].toFixed(2))*100;
  }

  // accuracy.push(confusion_matrix[TeachableMobileNet.classes_names.length*i+ i]/TeachableMobileNet.numValidation)
}

console.log("Confusion matrix as a matrix")
console.log(confusion_matrix_recipient);

return accuracy.map((elem: any)=>elem.toFixed(2)*100);
}

 
 async loadImages(number_of_species: number, classes_names: string[], options: object){
        TeachableMobileNet.classes_names=classes_names;
    
        await this.add_species(number_of_species, options);        

}

async add_species(number_of_species: number, options: object){
  
    //Loading feature model, used to create features from images
    //  await this.loadFeatureModel();

    for (let i = 0; i < TeachableMobileNet.classes_names.length; i++) {
      // this.add_images(this.classes_names[i], number_of_species, options);
    }
}

/**
 * 
 * @param name - name of the class receiving an example
 * @param number_of_species - how many images to add
 * @param options - details on the location of the images
 */
async add_images(name: string, number_of_species: number, options: any){   

    const class_add: any= [];

    for (let i = 0; i < number_of_species; i++) {      
    //   class_add.push(`${options.base}/${name}/${options.file_name} ${i}.${options.file_extension}`);
        
        //Uploading images
        const cake = new Image();        
        // cake.src = `${options.base}/${name}/${options.file_name} ${i}.${options.file_extension}`;
        cake.height=224;
        cake.width=224;
        cake.src="./assets/dataset/Can%C3%A1rio-da-Terra/image%200.jpeg"
        // console.log("Image location: ", cake.src )

        await new Promise<void>((resolve, reject) => {
            cake.onload = () => {

                //Finding the correspondent index of the class with name given
                const index= TeachableMobileNet.classes_names.findIndex((elem)=>elem===name)

                // this.addExample(index, cake);

              resolve();
            };
  
            cake.onerror = (error) => {
              // Handle error if the image fails to load
              reject(error);
            };
          });
  }

  // this.classes.push({name: name, images: class_add})  

  }



      
/**
 * This method will transform images into tensors
 * @param number_of_classes - number of classes
 * @param classes_names - name of each class
 */
async createTensors(number_of_classes: number, classes_names: string[]) {

    let output: any = [];

    /** There is a function on TensorFlow.js that also does that */
    const signatures= new Util().identityMatrix(number_of_classes);

    for (let i = 0; i < number_of_classes; i++) {

        this.classes[i].signature=signatures[i];
        this.classes[i].name=classes_names[i];

        for (let j = 0; j < this.classes[i].images.length; j++) {


        }

    }
}

    /**
     * Add a sample of data under the provided className
     * @param className the classification this example belongs to
     * @param sample the image / tensor that belongs in this classification
     */
    // public async addExample(className: number, sample: HTMLCanvasElement | tf.Tensor) {
public static async addExample(className: number, name: string, sample: HTMLImageElement | HTMLCanvasElement | tf.Tensor) {

            // console.log("Adding a new example...") 
          
            const cap = isTensor(sample) ? sample : capture(sample);
            
            
            //Getting the features
            const example = this.truncatedModel.predict(cap) as tf.Tensor;
            // console.log("Shape after feature extraction: ", example.shape)          
           

            const activation = example.dataSync() as Float32Array;
            
            //Very important to clean the memory aftermath, it makes the difference
            cap.dispose();
            example.dispose();
    
            // //Accessing the instance variable, not the local ones
            // // save samples of each class separately 
            
            if(!TeachableMobileNet.examples[className])
              //and an empty array, make sure there is not empty elements. 
              //it will create issue when transforming to tensors
              TeachableMobileNet.examples[className]=[]


              if(!TeachableMobileNet.classes_names[className])
                //Saving the lable when it first appears
                TeachableMobileNet.classes_names[className]=name;

            TeachableMobileNet.examples[className].push(activation);
    
            // // increase our sample counter
            TeachableMobileNet.totalSamples++;
        }


   /**
     * process the current examples provided to calculate labels and format
     * into proper tf.data.Dataset
     */
   static prepare() {
    for (const classes in TeachableMobileNet.examples){
        if (classes.length === 0) {
            throw new Error('Add some examples before training');
        }
    }

    const datasets: any = this.convertToTfDataset();

    this.trainDataset = datasets.trainDataset;
    this.validationDataset = datasets.validationDataset;
}

public prepareDataset() {
  for (let i = 0; i < TeachableMobileNet.numClasses; i++) {
    //Different from the original implementation of TM, mine is using example as static.
    //The goal is saving memory by using a single instance of the variable
    TeachableMobileNet.examples[i] = [];
  }
}

 
    // Optional seed to make shuffling of data predictable
    static seed: seedrandom.prng;



    
/**
     * Process the examples by first shuffling randomly per class, then adding
     * one-hot labels, then splitting into training/validation datsets, and finally
     * sorting one last time
     */
 
static convertToTfDataset() {

         // first shuffle each class individually
        // TODO: we could basically replicate this by insterting randomly
        for (let i = 0; i < TeachableMobileNet.examples.length; i++) {
          TeachableMobileNet.examples[i] = fisherYates(TeachableMobileNet.examples[i], this.seed) as Float32Array[];
      }

      // then break into validation and test datasets
      let trainDataset: Sample[] = [];
      let validationDataset: Sample[] = [];

      // for each class, add samples to train and validation dataset
      for (let i = 0; i < TeachableMobileNet.examples.length; i++) {
         
        // console.log("Number of classes: ", TeachableMobileNet.classes_names.length);

        const y = flatOneHot(i, TeachableMobileNet.classes_names.length);

        const classLength = TeachableMobileNet.examples[i].length;

        // console.log("Number of elements per class: ", classLength);

        const numValidation = Math.ceil(VALIDATION_FRACTION * classLength);
        const numTrain = classLength - numValidation;

        this.numValidation=numValidation;
       
        /**It is visiting per class, thus, it is possible to fix y, the target label */
     
      const classTrain = this.examples[i].slice(0, numTrain).map((dataArray) => {
          return { data: dataArray, label: y };
      });

      const classValidation = this.examples[i].slice(numTrain).map((dataArray) => {
          return { data: dataArray, label: y };
      });

      trainDataset = trainDataset.concat(classTrain);
      validationDataset = validationDataset.concat(classValidation); 


      }

      // console.log("Training element: ", trainDataset[trainDataset.length-1])      
      // console.log("Training length: ", trainDataset.length)
      // console.log("validation length: ", validationDataset.length);

      
        // finally shuffle both train and validation datasets
        trainDataset = fisherYates(trainDataset, this.seed) as Sample[];
        validationDataset = fisherYates(validationDataset, this.seed) as Sample[];

        const trainX = tf.data.array(trainDataset.map(sample => sample.data));
        const validationX = tf.data.array(validationDataset.map(sample => sample.data));
        const trainY = tf.data.array(trainDataset.map(sample => sample.label));
        const validationY = tf.data.array(validationDataset.map(sample => sample.label));

                // return tf.data dataset objects
                return {
                  trainDataset: tf.data.zip({ xs: trainX,  ys: trainY}),
                  validationDataset: tf.data.zip({ xs: validationX,  ys: validationY})
              };


}

datasetForEvaluation(){

}

/**Metrics */

static feature_aux: any;
static target_aux: any;

async evaluate(){
   
  if(!TeachableMobileNet.feature_aux){
    const features: any=[];
  const targets: any=[];

  for (let i = 0; i < TeachableMobileNet.examples.length; i++) {

    const y = flatOneHot(i, TeachableMobileNet.classes_names.length);
    


    //For class i, push all the examples.
    TeachableMobileNet.examples[i].forEach((elemn)=>{

       //Pushing the target signature
       targets.push(y);
       
       //Pushing features
       features.push(elemn)
    })

  }  

  TeachableMobileNet.feature_aux= tf.tensor(features);
  TeachableMobileNet.target_aux= tf.tensor(targets);

  } 

  const aux: any = this.trainingModel.evaluate(TeachableMobileNet.feature_aux, TeachableMobileNet.target_aux);
  
  return aux[1].dataSync()[0];


}


// async evaluate(){

//   const features: any=[];
//   const targets: any=[];

//   for (let i = 0; i < TeachableMobileNet.examples.length; i++) {

//     const y = flatOneHot(i, TeachableMobileNet.classes_names.length);
    


//     //For class i, push all the examples.
//     TeachableMobileNet.examples[i].forEach((elemn)=>{

//        //Pushing the target signature
//        targets.push(y);
       
//        //Pushing features
//        features.push(elemn)
//     })

//   }  

//   const aux_features= tf.tensor(features);
//   const aux_target= tf.tensor(targets);

  
//   // console.log("Tensor stack for evaluation: ", aux_features.shape)

//   const aux: any = this.trainingModel.evaluate(aux_features, aux_target);


//   return aux[1].dataSync()[0];


// }


/*** Final statistics */
/* 
     * Calculate each class accuracy using the validation dataset
*/
public async calculateAccuracyPerClass(validationData: any) {

  const validationXs = TeachableMobileNet.validationDataset.mapAsync(async (dataset: TensorContainer) => {
    return (dataset as { xs: TensorContainer, ys: TensorContainer}).xs;
});

const validationYs = TeachableMobileNet.validationDataset.mapAsync(async (dataset: TensorContainer) => {
    return (dataset as { xs: TensorContainer, ys: TensorContainer}).ys;
});

// console.log("validation dataset: ", validationXs);
 
// console.log("For calculating batch size: ", validationYs);

// we need to split our validation data into batches in case it is too large to fit in memory
const batchSize = Math.min(validationYs.size, 32);
// const batchSize =1;

const iterations = Math.ceil(validationYs.size / batchSize);

// console.log("Batch size: ", batchSize);

const batchesX = validationXs.batch(batchSize);

const batchesY = validationYs.batch(batchSize);
const itX = await batchesX.iterator();
const itY = await batchesY.iterator();
const allX = [];
const allY = [];

for (let i = 0; i < iterations; i++) {

    // 1. get the prediction values in batches
     const batchedXTensor = await itX.next();  

    //  console.log("Batch size on accuracy per class: ", batchedXTensor.value.shape);

    const batchedXPredictionTensor = this.trainingModel.predict(batchedXTensor.value) as tf.Tensor;

    const argMaxX = batchedXPredictionTensor.argMax(1); // Returns the indices of the max values along an axis

    allX.push(argMaxX);

    // 2. get the ground truth label values in batches
    const batchedYTensor = await itY.next();
    const argMaxY = batchedYTensor.value.argMax(1); // Returns the indices of the max values along an axis

    allY.push(argMaxY);
    
    // 3. dispose of all our tensors
     batchedXTensor.value.dispose();
     batchedXPredictionTensor.dispose();
     batchedYTensor.value.dispose();
}

      // concatenate all the results of the batches
      const reference = tf.concat(allY); // this is the ground truth
      const predictions = tf.concat(allX); // this is the prediction our model is guessing
      
      // console.log("this is the ground truth: ",  reference.dataSync())
      // console.log("This is the prediction our model is guessing: ",  predictions.dataSync())

        // only if we concatenated more than one tensor for preference and reference
        if (iterations !== 1) {
            for (let i = 0; i < allX.length; i++) {
                allX[i].dispose();
                allY[i].dispose();
            }
        }

    //  console.log("Lengtth: ", await reference.dataSync().length)  
    
    // const accuracyperclass=[];

    // const reference_aux= await reference.dataSync();
    // const prediction_aux= await predictions.dataSync();
    // console.log( predictions.dataSync());

    // reference_aux.forEach((element, index) => {
    //   if()
      
    // });  


        return { reference, predictions };  

}

}//end of class





/***Support methods (helpers) */

const isTensor = (c: any): c is tf.Tensor =>
    typeof c.dataId === 'object' && typeof c.shape === 'object';
/**
 * Converts an integer into its one-hot representation and returns
 * the data as a JS Array.
 */
function flatOneHot(label: number, numClasses: number) {

  const labelOneHot = new Array(numClasses).fill(0) as number[];
  labelOneHot[label] = 1;

  return labelOneHot;
}

interface Sample {
  data: Float32Array;
  label: number[];
}


/**
 * Shuffle an array of Float32Array or Samples using Fisher-Yates algorithm
 * Takes an optional seed value to make shuffling predictable
 */
function fisherYates(array: Float32Array[] | Sample[], seed?: seedrandom.prng) {
  const length = array.length;

  // need to clone array or we'd be editing original as we goo
  const shuffled = array.slice();

  for (let i = (length - 1); i > 0; i -= 1) {
      let randomIndex ;
      if (seed) {
          randomIndex = Math.floor(seed() * (i + 1));
      }
      else {
          randomIndex = Math.floor(Math.random() * (i + 1));
      }

      [shuffled[i], shuffled[randomIndex]] = [shuffled[randomIndex],shuffled[i]];
  }

  return shuffled;
}

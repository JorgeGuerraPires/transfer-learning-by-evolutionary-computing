
import {Class} from "./class"
import * as tf from '@tensorflow/tfjs';


export class Util {

/**
 * Receives an image and normalizes it between -1 and 1.
 * Returns a batched image (1 - element batch) of shape [1, w, h, c]
 * @param rasterElement the element with pixels to convert to a Tensor
 * @param grayscale optinal flag that changes the crop to [1, w, h, 1]
 */
capture(rasterElement: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement, grayscale?: boolean) {
    return tf.tidy(() => {
        const pixels = tf.browser.fromPixels(rasterElement);

        // // crop the image so we're using the center square
        const cropped = this.cropTensor(pixels, grayscale);

        // // Expand the outer most dimension so we have a batch size of 1
        const batchedImage = cropped.expandDims(0);

        // // Normalize the image between -1 and a1. The image comes in between 0-255
        // // so we divide by 127 and subtract 1.
        return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));    
    });
}    


cropTensor( img: tf.Tensor3D, grayscaleModel?: boolean, grayscaleInput?: boolean ) : tf.Tensor3D {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - (size / 2);
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - (size / 2);
    
    if (grayscaleModel && !grayscaleInput) {
        //cropped rgb data
        let grayscale_cropped = img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
        
        grayscale_cropped = grayscale_cropped.reshape([size * size, 1, 3])
        const rgb_weights = [0.2989, 0.5870, 0.1140]
        grayscale_cropped = tf.mul(grayscale_cropped, rgb_weights)
        grayscale_cropped = grayscale_cropped.reshape([size, size, 3]);
    
        grayscale_cropped = tf.sum(grayscale_cropped, -1)
        grayscale_cropped = tf.expandDims(grayscale_cropped, -1)

        return grayscale_cropped;
    }
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
}




/**
 * This function will make a copy of a model on the weight level
 * This is an attempt to avoid influencing the new mode when the old one
 * is eliminated. 
 * 
 * @param originalModel - the model to be copied 
 * @param recipient - the new model
 */
async copyModel_v3 (originalModel: tf.Sequential, recipient: tf.Sequential)  {

        originalModel.layers.forEach((layer, index)=>{
            recipient.layers[index].setWeights(layer.getWeights())
        })

    // originalModel.dispose();
}
    
/**
 * This function will make a copy of a TFJS model, as so it would be possible 
 * to erase the original.
 * @param model - model to be copied
 * @returns - copy of the model
 */  
async copyModel_v2 (originalModel: tf.Sequential)  {
        // Serialize the original model
        const modelTopology = originalModel.toJSON();

        // Load the serialized model into a new model
        const copiedModel = await tf.loadLayersModel(
            tf.io.fromMemory(modelTopology, undefined, undefined)
        );
    
        // Compile the copied model with the same settings as the original
        copiedModel.compile({
            loss: originalModel.loss,
            optimizer: originalModel.optimizer
        });
    
        return copiedModel;
}

/**
 * This function will make a copy of a TFJS model, as so it would be possible 
 * to erase the original.
 * @param model - model to be copied
 * @returns - copy of the model
 */  
copyModel (model: tf.Sequential)  {
    
        const copy = tf.sequential();`
        `
        model.layers.forEach(layer => {
            const aux =layer;
            // layer.dispose();
            copy.add(aux);
        });
        copy.compile({
            loss: model.loss,
            optimizer: model.optimizer
        });
        return copy;
    }


    removeElementByIndex(arr: any, index: number): number[] {
        // Check if the index is within bounds
        if (index >= 0 && index < arr.length) {
            // Remove the element at the specified index
            arr.splice(index, 1);            
        }
        return arr;
    }

    removeElement(arr: any, element: any): number[] {
        // Remove all occurrences of the specified element from the array
        return arr.filter((item: any) => item !== element);
    }

clean_array_of_tensors(tensors: tf.Sequential[]) {

    tensors.forEach((elem, index)=>{ 
        // if(!index_selection.includes(index))  
            elem.dispose() 
        });

}

 getClassNameBySignature(classes: Class[], signature: number[]) {           
    
        const class_name = classes.find(p => {
            let match = true;
            p.signature?.forEach ((elem,index)=>{ 
                if(elem!==signature[index])
                    match=false;                
            })

            return match;
        });

        return class_name ? class_name.name : "not found";
}

identityMatrix(n: number): number[][] {
        return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
}  


indexOfMax(arr: number[]): number {
        if (arr.length === 0) {
            return -1; // Return -1 for an empty array
        }
    
        let max = arr[0];
        let maxIndex = 0;
    
        for (let i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                maxIndex = i;
                max = arr[i];
            }
        }
    
        return maxIndex;
    }


suffle(array1:any, array2: any){
        // Shuffle the order of elements
    for (let i = array1.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));

    // Swap elements in both arrays
    [array1[i], array1[j]] = [array1[j], array1[i]];
    [array2[i], array2[j]] = [array2[j], array2[i]];
      }
    
    
    }

    sortByValuePreservingIndex<T>(
        arr1: number[],
        arr2: any[]
    ): T[] {

        // console.log("Vector for organizing: ", arr1)
        // arr2[0].summary()

        // Create an array of objects with value, index from arr1, and original index
        const pairingArray = arr1.map((value, index) => ({
            value,
            index,
            originalIndex: index,
            elementFromArr2: arr2[index], // Preserve the corresponding element from arr2
        }));
    
        // Sort the pairing array by value (largest to smallest)
        pairingArray.sort((a, b) => b.value - a.value);

        
    
        // Extract the sorted elements from arr2 based on the original index
        const sortedElementsFromArr2 = pairingArray.map(pair => pair.elementFromArr2);
    
        return sortedElementsFromArr2;
    }    
    
}


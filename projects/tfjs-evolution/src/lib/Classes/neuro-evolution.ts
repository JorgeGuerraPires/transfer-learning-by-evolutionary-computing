
import {TeachableMobileNet} from "../models/teachable-evolution"
import * as tf from '@tensorflow/tfjs';

/**
 * Each individual of the population will be made of 
 * an array, correspodent to the neural network weights and
 * an evaluation, its respective accuracy on the dataset given.
 * 
 */
interface Individual {
    individual: TeachableMobileNet,
    accuracy?: number
}

export abstract class NeuroEvolution {

        /**     
    This is the population. I have decided to create a population of just weights.
    I had to change my mind. Working with weights is tricky. 
    I have decided to sacrifice memory to make my life easiser.
    Tensors has their inner methods, it helps on the operators.
    I have to pay attention to memory manegement.
    Working with memory in tensors is tricky.     
     */

    population: Individual[]=[];
    private _population_size: number=30;
    
    /**Setters and Getters */
    public set population_size(population_size: number){
        this._population_size=population_size;
    }
    public get population_size(){
        return this._population_size;
    }

  /**
   * Methods with implementation
   */

   async evaluate(){
     return this.population.forEach(async (elem)=>{
        elem.accuracy= await elem.individual.evaluate();        
    })
  }

    start_population(){

        for(let i=0; i<this.population_size;i++){            
            this.population.push({individual: new TeachableMobileNet()})
        }
        // console.log("Population length: ", this.population.length);
        // console.log("Memory after population iniialization: ", tf.memory().numBytes);
    }

    async apply_evolution(iteration: number){

        for(let i=0;i<iteration;i++){
            
            console.log("Interation: ",i)
            console.log("Memory: ", tf.memory().numBytes);
            console.log("Tensors in memory: ", tf.memory().numTensors);
            console.log("Population length: ", this.population.length);
            await this.evolution();
        }

    }
 
    /**
     * This method will apply the evolution
     */
    async evolution(){

        
        
        //Update the accuracy
        await this.evaluate();

        const offspring=[];

        /**First step is crossover, for creating offspring */
        for(let i=0; i<this.population.length;i++)
            {
                //The population will double in size
                offspring.push(this.crossover()); 
            }

       //Mutation on the offspring
    //    console.log("Weight before mutation: ", offspring[0].individual.getHead().layers[0].getWeights()[0].dataSync());

       this.mutation(offspring)   

        //Joining parents and offspring into the same pool for selection
        this.population=[...this.population, ...offspring];

        //Update the accuracy
         await this.evaluate();


         //selection
         this.selection();
        console.log("Best accuracy: ", this.getTheBest().accuracy)
        console.log("Worst accuracy: ", this.getTheWorst().accuracy)        

    }

    /**Abstract methods.
     * Should be implemented by each variation of evolutionary algorithms
     */

    abstract crossover(): Individual;
    abstract mutation(offspring: Individual[]): void;
    abstract selection(): void;


    /**Helpers */

    getTheBest(): Individual{
        let index_best: number=0;
        let best=0;

        this.population.forEach((elem: any, index: number)=>{

            if(elem.accuracy>best){
                best=elem.accuracy;
                index_best=index
            }
                
        })

        return this.population[index_best];
    }


    getTheWorst(): Individual{
        let index_worst: number=0;
        let worst=1;

        this.population.forEach((elem: any, index: number)=>{

            if(elem.accuracy<worst){
                worst=elem.accuracy;
                index_worst=index
            }
                
        })

        return this.population[index_worst];
    }

        /**
     * Selecting individuals by elitism
     */
    select_parents_by_rollet_wheels_v2(selection_number=2): Individual[]{

            // console.log("Selecting parents by rollet wheels...");
    
            const share: any=[];
    
            let total: any=0;
              
            // console.log("Population length...", this.population.length);
    
            //This will create an overall accuracy
            this.population.forEach((elem)=>total+=elem.accuracy);
    
            let pointer=0;
    
            this.population.forEach((elem:any, index)=>{
    
                // console.log("Accuracy...", elem.accuracy);
    
                share.push(
                    {
                      index: index,
                      share:  elem.accuracy/total,
                      min: pointer,
                      max: pointer + (elem.accuracy/total)
                  })
    
                  pointer+=elem.accuracy/total;
            });
    
            // console.log("Shares: ", share);
    
            const aux_selection: any =[];
    
              //Selecting two individuals using roulette wheels
            let counter=0;

            const indexes_aux: number[]=[];

            while(counter<selection_number){
                
                //Random pointer, used to randomly select an individual
                const aux_index= Math.random();
    
                share.forEach((elem: any)=>{
    
                    if((elem.min<aux_index)&&(elem.max>aux_index)){
                        
                        const aux: Individual = {individual: new TeachableMobileNet(), accuracy: this.population[elem.index].accuracy }
                        
                        // console.log("Individual just created (weigth): ", aux.individual.getHead().layers[0].getWeights()[0].dataSync());
                        
                        // console.log("Individual original: ", this.population[elem.index].individual.getHead().layers[0].getWeights()[0].dataSync());
                        //               // Original                                        Recipient
                        // this.copyModel(this.population[elem.index].individual.getHead(),aux.individual.getHead())

                        // this.population[elem.index].individual.getHead().dispose();
                        
                        // console.log("Individual just created (weigtht, copied): ", aux.individual.getHead().layers[0].getWeights()[0].dataSync());
                                               

                        counter++;
                        aux_selection.push(this.population[elem.index]);

                        // indexes_aux.push(elem.index)
                    }
                })
            }

            // this.population.forEach((elem, index)=>{
            //     if(!indexes_aux.includes(index)){
            //         console.log("Disposed...")
            //         elem.individual.getHead().dispose();
            //     }
                    
            // })

            // this.population=[];

            return aux_selection;
        }
    
    /**
     * Selecting individuals by elitism
     */
    select_parents_by_rollet_wheels(selection_number=2): Individual[]{

        // console.log
        // ("Selecting parents by rollet wheels...");

        const share: any=[];

        let total: any=0;
          
        // console.log("Population length...", this.population.length);

        //This will create an overall accuracy
        this.population.forEach((elem)=>total+=elem.accuracy);

        let pointer=0;

        this.population.forEach((elem:any, index)=>{

            // console.log("Accuracy...", elem.accuracy);

            share.push(
                {
                  index: index,
                  share:  elem.accuracy/total,
                  min: pointer,
                  max: pointer + (elem.accuracy/total)
              })

              pointer+=elem.accuracy/total;
        });

        // console.log("Shares: ", share);

        const aux_selection: any =[];

          //Selecting two individuals using roulette wheels
        let counter=0;
        while(counter<selection_number){
            
            //Random pointer, used to randomly select an individual
            const aux_index= Math.random();

            share.forEach((elem: any)=>{

                if((elem.min<aux_index)&&(elem.max>aux_index)){
                    counter++;
                    aux_selection.push(this.population[elem.index]);
                }
            })
        }
        return aux_selection;
    }

/**
 * This function will make a copy of a model on the weight level
 * This is an attempt to avoid influencing the new model when the old one
 * is eliminated. 
 * 
 * @param originalModel - the model to be copied 
 * @param recipient - the new model
 */
async copyModel (originalModel: tf.LayersModel, recipient: tf.LayersModel)  {

    originalModel.layers.forEach((layer: any, index: any)=>{
        recipient.layers[index].setWeights(layer.getWeights())
    })
}    

}

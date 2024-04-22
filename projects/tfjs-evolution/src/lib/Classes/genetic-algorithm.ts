import { NeuroEvolution } from "./neuro-evolution";
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


export class GeneticAlgorithm extends NeuroEvolution { 
    
    

    /**
     * Implementatin abstract methods
     */

    override crossover(): Individual {
        // console.log("Applying crossover on Genetic Algorithms...")

        const selected = this.select_parents_by_rollet_wheels();

        const individual_1 = selected[0];
        const individual_2 = selected[1];

        // console.log("Individual 1: ")
        // individual_1.individual.getHead().summary();

        let individual_3: Individual= {individual: new TeachableMobileNet};

        const alfa = Math.random();

        individual_3.individual.getHead().layers.forEach((layer, index)=>{
            individual_3.individual.getHead().layers[index].setWeights(
                individual_1.individual.getHead().layers[index].getWeights()
                .map((elem: any, index_2: any)=> elem.mul(alfa)
                .add(individual_2.individual.getHead().layers[index]
                .getWeights()[index_2]
                .mul(1 - alfa)
            )))
        })

        // this.population.push(individual_3);
        // console.log("Population: ", this.population.length);  
        
        return individual_3;        
    
    }//end of crossover

    override mutation(offspring: Individual[]): void {

        const multation_rate=1;

        offspring.forEach((elem)=>{
            elem.individual.getHead().setWeights(
                elem.individual.getHead().getWeights().map((elem_2)=> {               
                  return elem_2.add(elem_2.mul(multation_rate*Math.random()))     
                })
              );
        })
    }

    override selection(): void {

        const selected_individuals= this.select_parents_by_rollet_wheels(this.population_size);

        // const selected_individuals_copy: any=[];

        // selected_individuals.forEach(async (elem)=>{

        //     const aux: Individual= {individual: new TeachableMobileNet()};

        //     // await this.copyModel(elem.individual.getHead(), aux.individual.getHead());

        //     // aux.individual.getHead().summary();

        //     selected_individuals_copy.push({individual: aux, accuracy: elem.accuracy});
        // });

        
        // console.log("Memory before cleaning: ", tf.memory().numBytes);
        
        //Cleaning memory
        // this.population.forEach((elem)=> elem.individual.getHead().dispose())
        // this.population=[];

        //The best will enter on the population by elitism
        const best= this.getTheBest();

        this.population=[];

        this.population=selected_individuals;

        //Adding one extra individual by elitism         
        // this.population.push(best);

        console.log("Best accuracy (elitism): ", best.accuracy);
        
        
        // console.log("Current population length: ", this.population.length);
        // console.log("Current population: ", this.population);


        // console.log("After selection: ", this.population[0])
        // console.log("Memory before cleaning: ", tf.memory().numBytes);
    }
}

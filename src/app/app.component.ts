import { AfterViewInit, Component, ViewChild  } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet } from '@angular/router';
import { Group } from './interfaces/group';
import * as tf from '@tensorflow/tfjs';

//Importing from the current package, not published yet, local version
// import { DisplayPanelComponent } from '../../projects/tfjs-evolution/src/public-api';
// import { TeachableMobileNet } from '../../projects/tfjs-evolution/src/lib/models/teachable-evolution';
// import { GeneticAlgorithm } from '../../projects/tfjs-evolution/src/lib/Classes/genetic-algorithm';


//Import from the package in NPM
import { DisplayPanelComponent, TeachableMobileNet, GeneticAlgorithm } from 'tfjs-evolution';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, RouterOutlet, DisplayPanelComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent  implements AfterViewInit{
  classes_names=[
    "Canário-da-Terra",
    "Pardal",
    "Bem-Te-Vi",
    "Suiriri-Cavaleiro",
    "Lavadeira-Mascarada",
    "Anu-Branco",  
    "Beija-Flor-Tesoura", 
    "Casaca-de-Couro-da-Lama",
    "Garibaldi",
    "Periquitão",
    "Pombo-Doméstico",
    "Quero-Quero",
    "Rolinha-Roxa",
    "Sabiá-Do-Campo",
    "Urubu-de-Cabeça-Preta"
  ]

  confusion_matrix: [][]=[]

  accuracy_per_class: number[]=[];



  classes!: Group[];

  @ViewChild(DisplayPanelComponent) child!:DisplayPanelComponent;

  constructor(){
    
  }

  ngAfterViewInit(): void {    

    // this.main();
    this.main_v2();

  }

loadImage(){
  
  //All the population share the same features
  const options={ 
    base: "./assets/dataset",//base where are you iamges
    file_name:"image",//how your images are named
    file_extension:"jpeg" //extension used
  };

  //Load images
  /**The TensorFlow.js memory is not changing with more images, it is good. 
   * It is possible to load more images, without increasing the risk of memory leaking
   * 
  */
  this.child.loadImages(20, this.classes_names, options); 

}

async train() {

  //Loading feature model
  await TeachableMobileNet.loadFeatureModel();

 //Adding examples to the Teachable Machine.
 //There is a small delay on the function, for waiting for the images to load. 
 await this.child.addexamples();

 //Starting the population, the evolution part
 const population = new GeneticAlgorithm();

  //Starting population
  population.start_population();

  //Applying evolution
  population.population_size=40;  
  await population.apply_evolution(5);

  //Getting the best
  const best= population.getTheBest();

  const worst= population.getTheWorst();



  // console.log("Best accuracy: ", best.accuracy)
  await best.individual.train();

  this.accuracy_per_class= await best.individual.accuracy_per_class(this.confusion_matrix);

}

 async main_v2(){

 


 } 

 async main(){ 

    // const teachablemachine = new TeachableMobileNet();

    const options={ 
      base: "./assets/dataset",//base where are you iamges
      file_name:"image",//how your images are named
      file_extension:"jpeg" //extension used
    };


    console.log("TFJS memory (starting point): ", tf.memory().numBytes);

    // await teachablemachine.loadImages(1, this.classes_names, options);

    console.log("TFJS memory (after loading images): ", tf.memory().numBytes);

    
    //Just waiting for images to upload


    //Finally, call the load images methods
    /**Improvement on memory management. the previous version was dependent on the image 
     * number. Now, it is not dependent anymore.
     * I have used insights from the Teachable Machine repository.
     * This is an effective trick to avoid creating memory garbage as we go
     */
    this.child.loadImages(23, this.classes_names, options); 

    //Loading feature model
    await TeachableMobileNet.loadFeatureModel();

    //Adding examples to the Teachable Machine.
    //There is a small delay on the function, for waiting for the images to load. 
    await this.child.addexamples();

    console.log("TFJS memory (after adding images ): ", tf.memory().numBytes);

    //Loading feature model
    // TeachableMobileNet.createHead();
    const model = new TeachableMobileNet();
    model.createHead();

    TeachableMobileNet.convertToTfDataset();

    model.train();

    
    

  }


}

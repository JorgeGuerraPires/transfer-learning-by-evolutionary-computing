import { Component} from '@angular/core';
import { Group } from '../../interfaces/group';
import { CommonModule } from '@angular/common';
import { TeachableMobileNet } from '../../models/teachable-evolution';

@Component({
  selector: 'neuroevolution-display-panel',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './display-panel.component.html',
  styleUrl: './display-panel.component.css'
})
export class DisplayPanelComponent {

  classes: Group[]=[];

  classes_names!: string[];

  number_of_samples_per_class!: number;

loadImages(number_of_species: number, classes_names: string[], options: object){
    
    this.classes_names=classes_names;
    this.number_of_samples_per_class=number_of_species;

    this.add_species(number_of_species, options);    

  }


add_species(number_of_species: number, options: object){

    for (let i = 0; i < this.classes_names.length; i++) {
      this.add_images(this.classes_names[i], number_of_species, options);
      
    }

  }

  add_images(name: string, number_of_species: number, options: any){   

    const class_add: any= [];

    for (let i = 0; i < number_of_species; i++) {      
      class_add.push(`${options.base}/${name}/${options.file_name} ${i+1}.${options.file_extension}`);
  }

  this.classes.push({name: name, images: class_add})  

  }

 async addexamples(){

  //This is needed to make sure it gives time for the images to upload
  //The images upload very fast, what makes this method execute before the images are on HTML
  //It can be removed if somehow this method is just called after the images are available.
  // console.log("Loading examples as tensors....")
  await this.delay(0);

  for (let i = 0; i < this.classes_names.length; i++) {
    await this.add_example(this.classes_names[i], this.number_of_samples_per_class);
  }
  }

  async add_example(name: string, number_of_species: number){   

    const class_add: any= [];
    // console.log(name)
    for (let i = 0; i < number_of_species; i++) { 
      
      //Collecting the images from HTML
      const aux = document.getElementById(`class-${name}-${i}`) as HTMLImageElement;      

      //Adding the example
      const index= this.classes_names.findIndex((elem)=>elem===name)

      await TeachableMobileNet.addExample(index, name, aux);    
      
  }

  // this.classes.push({name: name, images: class_add})  

  }

  delay(ms: number) {
    return new Promise<void>((resolve) => setTimeout(resolve, ms));
}
}

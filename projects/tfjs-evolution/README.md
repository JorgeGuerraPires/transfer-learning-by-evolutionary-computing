# Tfjs Evolution

This is a placeholder for a package I am planning to publish.
I have experimented with Evolutionary Computing in TensorFlow.js, and I am planning a paper and a NPN package.
I was unable to find a NPM package, and decided to create one!

I am releasing the parts by parts. 
You can start to test the released parts, and let me know your thoughts.

# Seeing your images

You can see your images, it can useful to have an idea about how it looks like, what you are trying to feed the model.
Remember: "garbage in garbage out".

Import the directive for the panel.

`import { DisplayPanelComponent } from 'tfjs-evolution';`


Make sure to import the component. Angular is now standalone by default.

`  imports: [DisplayPanelComponent]`

Create a local instance, for calling the component method.

` @ViewChild(DisplayPanelComponent) child!:DisplayPanelComponent;`

Add a list of class names. It must be the same on the folder names.

```` typescript
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
````
Make the call for loading images. You could most likely make the call at the constructor.


```typescript
ngAfterViewInit(): void {
    const options={ 
      base: "./assets/dataset",//base where are your iamges
      file_name:"image",//how your images are named
      file_extension:"jpeg" //extension used
    };

    //Finally, call the load images methods
    this.child.loadImages(5, options)
  }
```

Finally, do not forget to add the HTML code

```html
<h1>Images used on the training</h1>
<neuroevolution-display-panel></neuroevolution-display-panel>

```

You should see in our HTML file, in Angular, the images


# Updates

I have finished! Hope to publish a paper soon!

<!-- ## Early results -->

<!-- ![alt text](./images/confusion%20matrix.png) -->

# Further help

Feel free to get in touch: jorgeguerrabrazil@gmail.com

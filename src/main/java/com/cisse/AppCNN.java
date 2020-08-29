package com.cisse;

import java.io.File;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer.PoolingType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;

public class AppCNN {

	private Logger slf4jLogger;
	
	public static void main(String[] args) throws Exception, Exception {
		// TODO Auto-generated method stub
		long height = 28;
		long width = 28;
		long depth = 1;  // profondeur de l'image , comme il est blanc noir nous auront 1 couche,en cas de couleur nous auront RVB dont 3 couches
		double learningRate = 0.001;
		long nombreFiltreDeConvolution = 20;
		int outputSize = 10; // car la sortie sont des chiffres de 1 a 10
		int batchSize = 100;
		int numberEpochs = 1; //Car nous avons beaucoup d'image
		int seed = 1234; 
		
		 
		System.out.println("------- CREATION DU MODEL -------------");
		MultiLayerConfiguration configDuModel = new NeuralNetConfiguration.Builder()
				            .seed(seed)
				            .updater(new Adam(learningRate)) //Vitesse d'apprentissage
				            .weightInit(WeightInit.XAVIER) // Algo de requilibre du poids des neurones
				            .list()
				            .setInputType(InputType.convolutionalFlat(height, width, depth)) //Nous ldonnons une image en Input
				            .layer(0,new ConvolutionLayer.Builder()
				            		.nIn(depth)
				            		.nOut(nombreFiltreDeConvolution)
				            		.activation(Activation.RELU) // appliquer la fonction d'activation relu afin d'avoir des zero ou des x dans notre tableau de pixel
				            		.kernelSize(5,5)  //Taille du filtre de convolution a appliquer 5pixel x 5 pixel
				            		.stride(1,1)  //Nombre de pas en appliquant le filtre	
				            		.build())
				          
				            .layer(1,new SubsamplingLayer.Builder()   //Pour le cas du MaxPooling pour garder que les maximums				           		
				            		.kernelSize(2,2)  // utiliser 4pixel
				            		.stride(2,2)  //glisser de 4 pixel 
				            		.poolingType(PoolingType.MAX)
				            		.build()) 
				            
				            .layer(2,new ConvolutionLayer.Builder()
				            		.nOut(50)
				            		.activation(Activation.RELU)
				            		.kernelSize(5,5)
				            		.stride(1,1)
				            		.build())
				          
				            .layer(3,new SubsamplingLayer.Builder() //Pour le cas du MaxPooling
				            		.kernelSize(2,2)  // utiliser 4pixel
				            		.stride(2,2)  //glisser de 4 pixel 
				            		.poolingType(PoolingType.MAX)
				            		.build()) 
				            
				            .layer(4,new DenseLayer.Builder()  // Ajouter a la fin une couche fully-connected
				            		.nOut(500)
				            		.activation(Activation.RELU)
				            		.build()) 
				            
				          
                            .layer(5,new OutputLayer.Builder()  //Couche  de sortie
                            		.nOut(outputSize)
                            		.activation(Activation.SOFTMAX) // softMax pour la sortie comme une probabilité de 1
                            		.lossFunction(LossFunction.NEGATIVELOGLIKELIHOOD) //Fonction de calcul d'erreur
                            		.build()) 
                            .build(); // pour construire la structure du model

	     MultiLayerNetwork model = new MultiLayerNetwork(configDuModel);
	     model.init();
	             
	            // System.out.println(configDuModel.toJson()); //Pour voir les parametres de sont model
	     
	             
	             

         	     //Suivit (Demarrage du serveur de monitoring du processus d'apprentissage) de l'apprentissage avec uiserver qui donne une vue graphique sur le port 9000
      System.out.println("----SUIVIT DE L'APPRENTISSAGE VIA DES GRAPHES AVEC ND4J-----");	     
    	     
      UIServer uiServer = UIServer.getInstance();
      InMemoryStatsStorage inMemoryStatsStorage = new InMemoryStatsStorage(); //loguer les informations durant l'apprentissage
    	    
      uiServer.attach(inMemoryStatsStorage); // lier uiserver et inMemoryStateStorage
      model.setListeners(new StatsListener(inMemoryStatsStorage));
    	
            
	             
    System.out.println("-------- ENTRAINEMENT DU MODEL ------------");
	
    File fileTrain = new ClassPathResource("training").getFile();	
	RecordReader recordReaderTrain = new ImageRecordReader(height,width,new ParentPathLabelGenerator());
	recordReaderTrain.initialize(new FileSplit(fileTrain,NativeImageLoader.ALLOWED_FORMATS,new Random(seed)));
	
	DataNormalization dataNormalization = new ImagePreProcessingScaler(0,1); //Normalisation des données en leur donnant tous des valeur min et max (0,1) pour eviter le phenomene de surapprentissage	
	DataSetIterator datasetIteratorTrain = new RecordReaderDataSetIterator(recordReaderTrain,batchSize,1,outputSize);
	
	datasetIteratorTrain.setPreProcessor(dataNormalization); // Pour appliquer la normalisation a notre DatasetTrain
	
	for (int i = 0; i < numberEpochs; i++) {
		model.fit(datasetIteratorTrain);
	}
	
	System.out.println("-----------------DEBUT EVALUATION DU MODEL ----------------");
	    File fileTest = new ClassPathResource("testing").getFile();	
		RecordReader recordReaderTest = new ImageRecordReader(height,width,new ParentPathLabelGenerator());
		recordReaderTest.initialize(new FileSplit(fileTest,NativeImageLoader.ALLOWED_FORMATS,new Random(seed)));
		
		DataNormalization dataNormalizationTest = new ImagePreProcessingScaler(0,1); //Normalisation des données en leur donnant tous des valeur min et max (0,1) pour eviter le phenomene de surapprentissage	
		DataSetIterator dataSetIteratorTest = new RecordReaderDataSetIterator(recordReaderTest,batchSize,1,outputSize);
		dataSetIteratorTest.setPreProcessor(dataNormalizationTest); // Pour appliquer la normalisation a notre DatasetTrain
		
		Evaluation evaluation = new Evaluation(); //Pour evaluer le model
	      
	      while (dataSetIteratorTest.hasNext()) {
			DataSet dataSetTest = dataSetIteratorTest.next(); //Parcourir le dataSet
			INDArray features = dataSetTest.getFeatures();   //Recuperer les entrees du dataset
			INDArray TargetLabels   = dataSetTest.getLabels();
			INDArray predictedLabels = model.output(features); // prediction de la sortie en fonction des entrees
			evaluation.eval(predictedLabels, TargetLabels);
			
		}
	      
	      System.out.println(evaluation.stats()); // Qui nous affiche une matrice de confusion
	      
	System.out.println("-------------------FIN EVALUATION DU MODEL -----------------");
	
	 
    // Enregistrement du model
    System.out.println("--------- ENREGISTREMENT  DU MODEL POUR UNE UTILISATION ULTERIEUR------------");
    ModelSerializer .writeModel(model, "AppCNNModel.zip", true); //enregistrer le model avec son algo d'apprentissage updater

	
	
	
	
	
	
	
	/*while (datasetIteratorTrain.hasNext()) {       //POur voir le contenu du model a entrainer
		DataSet dataset =  datasetIteratorTrain.next();
		INDArray features = dataset.getFeatures();
		INDArray labels = dataset.getLabels();
		System.out.println(features.shapeInfoToString());
		System.out.println(labels.shapeInfoToString());
		
		System.out.println("---------------------");
		
		
	}
	 */
	
	
   
	
	
	}

}

/*
 * To the extent possible under law, the ImageJ developers have waived
 * all copyright and related or neighboring rights to this tutorial code.
 *
 * See the CC0 1.0 Universal license for details:
 *     http://creativecommons.org/publicdomain/zero/1.0/
 */


package com.mycompany.imagej;
//package net.codejava.swing.jcheckbox;

import net.imagej.axis.Axes;
import net.imagej.axis.AxisType;
import net.imagej.Dataset;
import net.imagej.DatasetService;
import net.imagej.ImageJ;
import net.imagej.ops.OpService;
//import net.imagej.ImgPlus;
//import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.RealType;
//import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
//import org.apache.commons.lang.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.scijava.command.Command;
import org.scijava.ItemIO;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;

/*
import org.deeplearning4j.clustering.cluster.Cluster;
import org.deeplearning4j.clustering.cluster.ClusterSet;
import org.deeplearning4j.clustering.cluster.Point;
import org.deeplearning4j.clustering.cluster.PointClassification;
import org.deeplearning4j.clustering.kmeans.KMeansClustering;
*/
//import org.apache.commons.math4.geometry.partitioning.Region.Location;
import org.apache.commons.math4.ml.clustering.CentroidCluster;
import org.apache.commons.math4.ml.clustering.Clusterable;
//import org.apache.commons.math4.ml.clustering.Clusterer;
import org.apache.commons.math4.ml.clustering.KMeansPlusPlusClusterer;
//import org.apache.commons.math4.ml.neuralnet.twod.util.LocationFinder.Location;

//import java.util.concurrent.TimeUnit;
import java.io.File;
import java.text.ParseException;
import java.util.ArrayList;
//import java.util.Arrays;
import java.util.List;
//import java.util.stream.IntStream; 
import java.util.stream.IntStream;

//import java.awt.FlowLayout;
//import java.awt.event.ActionEvent;
//import java.awt.event.ActionListener;

import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JSpinner;
//import javax.swing.JLabel;
//import javax.swing.JTextField;
//import javax.swing.SwingUtilities;
import javax.swing.*;
import javax.swing.JPanel;
import javax.swing.BorderFactory;
import javax.swing.JOptionPane;
import java.awt.FlowLayout;
import java.lang.Object;


/**
 * This example illustrates how to create an ImageJ {@link Command} plugin.
 * <p>
 * The code here is a simple Gaussian blur using ImageJ Ops.
 * </p>
 * <p>
 * You should replace the parameter fields with your own inputs and outputs,
 * and replace the {@link run} method implementation with your own logic.
 * </p>
 */
@Plugin(type = Command.class, menuPath = "Plugins>LUMoS Spectral Unmixing")
public class LUMoS_Spectral_Unmixing<T extends RealType<T>> implements Command {
    //
    // Feel free to add more parameters here...
    //

	
    @Parameter(label="Number of fluorophores to unmix", style="spinner", min="1", max="1024")
    private int k;
    
    @Parameter(label="Manually set advanced options", style="checkbox")
    private boolean advanced_options;
    
    @Parameter(label="Remove background only (no unmixing)", style="checkbox")
    private boolean remove_background_only;
    
    /*
    @Parameter(label="Maximum number of iterations", style="spinner", min="-1", max="1024")
    private int max_iter;
    
    @Parameter(label="Number of replicates (runs k-means multiple times and uses best result)", style="spinner", min="1", max="1024")
    private int num_replicates;
    
    @Parameter(label="Representitive timestep", style="spinner", min="1", max="1024")
    private int representitive_timestep;
    
    @Parameter(label="Representitive slice", style="spinner", min="1", max="1024")
    private int representitive_slice;
    
    @Parameter(label="Unmix Based on Pixel Intensity Ratio (Instead of Absolute Channel Values)", style="checkbox")
    private boolean intensity_ratios;
    */
    

	@Parameter
	private DatasetService datasetService;
	
    @Parameter
    private Dataset currentData;

    @Parameter
    private UIService uiService;

    @Parameter
    private OpService opService;
    
    @Parameter(type = ItemIO.OUTPUT)
	private Dataset dataset;

    
    
    
    @Override
    public void run() {
    	

    	//Load Data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    	final Img<T> image = (Img<T>)currentData.getImgPlus();//Img<T> test_image = (Img<T>)currentData.getImgPlus().getImg();
    	long num_channels = currentData.getChannels();
    	k=k+1; //add cluster for background
    	long num_cols = currentData.getWidth();
        long num_rows = currentData.getHeight();
        long num_slices = currentData.getDepth();
        long num_timesteps = currentData.getFrames(); 
    	
        //Set up input variables~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    	ArrayList<Integer> selected_channels = new ArrayList<>(); //final int[] channels_to_unmix = {1, 2}; //nd4jindexes from 0
    	
    	
    	//myModel = new CyclingSpinnerListModel("test");
    	//JSpinner max_iter_spinner = new JSpinner(this, "test", myModel);
        JFrame.setDefaultLookAndFeelDecorated(true);
        JFrame frame = new JFrame("Simple checkbox demo");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        JOptionPane panel = new JOptionPane("");
        panel.setBorder(BorderFactory.createTitledBorder("Select Channels to Unmix")); 
        List<JCheckBox> checkboxes = new ArrayList<>();
        int param_index = 0;
        int representitive_timestep = (int)(num_timesteps/2)+1;
        int representitive_slice = (int)(num_slices/2)+1;
        int num_replicates = 10;
        int max_iter = 100;
        boolean intensity_ratios = true;
        
        

        
        //prompt for input
        
        
        
        JCheckBox checkbox = new JCheckBox(Integer.toString(1), true);
        for (int c=0; c<num_channels; c++) {
	        checkbox = new JCheckBox(Integer.toString(c+1), true);
	        checkboxes.add(checkbox);
	        panel.add(checkboxes.get(c));
        }
        String message = "Select Channels to Unmix";
        Object[] params = new Object[(int)num_channels+10];
        params[0] = message;
        for (int c=0; c<num_channels; c++){
        	checkbox = checkboxes.get(c);
        	params[c+1] = checkbox;
        	param_index = c+2;
        }

        
        JCheckBox ratio_checkbox = null;
        JPanel replicate_panel;
        JSpinner replicate_input = null;
        JPanel representitive_timestep_panel;
        JSpinner representitive_timestep_input = null;
        JPanel representitive_slice_panel;
        JSpinner representitive_slice_input = null;
        JPanel max_iter_panel;
        JSpinner max_iter_input = null;
        int[] channels_to_unmix = null;

        if  (remove_background_only) {
            channels_to_unmix = IntStream.rangeClosed(0, (int)num_channels-1).toArray();
            k = (int)num_channels+1; 
            num_replicates = 1;
        	
        } else {
        
	        if (advanced_options) {
	        	//Ask for number of replicates
	        	replicate_panel = new JPanel(new FlowLayout(FlowLayout.LEFT, 3,3));
	        	replicate_panel.add(new JLabel("Number of replicates (runs k-means multiple times and uses best result)"));
	        	params[param_index] = replicate_panel;
	        	param_index ++;
	        	
	        	replicate_input = new JSpinner(new SpinnerNumberModel(10,1,1024,1));
	        	params[param_index] = replicate_input;
	        	param_index ++;
	        	
	        	//Ask for representitive timestep
	        	representitive_timestep_panel = new JPanel(new FlowLayout(FlowLayout.LEFT, 3,3));
	        	representitive_timestep_panel.add(new JLabel("Representitive timestep"));
	        	params[param_index] = representitive_timestep_panel;
	        	param_index ++;
	        	
	        	representitive_timestep_input = new JSpinner(new SpinnerNumberModel((int)(num_timesteps/2)+1,1,num_timesteps,1));
	        	params[param_index] = representitive_timestep_input;
	        	param_index ++;
	        	
	        	//Ask for representitive slice
	        	representitive_slice_panel = new JPanel(new FlowLayout(FlowLayout.LEFT, 3,3));
	        	representitive_slice_panel.add(new JLabel("Representitive slice"));
	        	params[param_index] = representitive_slice_panel;
	        	param_index ++;
	        	
	        	representitive_slice_input = new JSpinner(new SpinnerNumberModel((int)(num_slices/2)+1,1,num_slices,1));
	        	params[param_index] = representitive_slice_input;
	        	param_index ++;
	        	
	        	//Ask for max number of iterations
	        	max_iter_panel = new JPanel(new FlowLayout(FlowLayout.LEFT, 3,3));
	        	max_iter_panel.add(new JLabel("Maximum number of iterations"));
	        	params[param_index] = max_iter_panel;
	        	param_index ++;
	        	
	        	max_iter_input = new JSpinner(new SpinnerNumberModel(100,1,1024,1));
	        	params[param_index] = max_iter_input;
	        	param_index ++;
	        	
	        	//Ask whether to use intensity ratios
	        	ratio_checkbox = new JCheckBox("Unmix Based on Pixel Intensity Ratio (Instead of Absolute Channel Values)", true);
	        	params[param_index] = ratio_checkbox;
	        	param_index ++;
	        	
	        }
	        
	        JOptionPane.showConfirmDialog(frame, params, "LUMoS Settings", JOptionPane.CANCEL_OPTION);
	
	        
	        for (int c=0; c<num_channels; c++) {
		        boolean checked = checkboxes.get(c).isSelected();
		        if (checked) {
		        	selected_channels.add(c);
		        }
	        }
	        
	        if (advanced_options){
	        	intensity_ratios = ratio_checkbox.isSelected();
			    try {
					representitive_timestep_input.commitEdit();
					representitive_slice_input.commitEdit();
					replicate_input.commitEdit();
					max_iter_input.commitEdit();
				} catch (ParseException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
	        	String value = representitive_timestep_input.getValue() + "";
	        	representitive_timestep = (int)Double.parseDouble(value);
	        	
	        	value = representitive_slice_input.getValue() + "";
	        	representitive_slice = (int)Double.parseDouble(value);
	
	        	value = representitive_slice_input.getValue() + "";
	        	representitive_slice = (int)Double.parseDouble(value);
	        	
	        	value = replicate_input.getValue() + "";
	        	num_replicates = (int)Double.parseDouble(value);
	        	
	        	value = max_iter_input.getValue() + "";
	        	max_iter = (int)Double.parseDouble(value);
	        }
	        
	        
	        channels_to_unmix = selected_channels.stream().mapToInt(i -> i).toArray();

        }
        
        
        
        //Get handle on shape and size of input image (re-order if necessary)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        long num_channels_to_unmix = channels_to_unmix.length;
        long num_channels_to_leave = num_channels - num_channels_to_unmix;

        
        if (representitive_slice<1 || representitive_slice>num_slices) {
        	JOptionPane.showMessageDialog(null, "Representitive slice must be between 1 and the number of slices. You selected "+representitive_slice
        			+ " and the maximum is " + num_slices, "Error", JOptionPane.ERROR_MESSAGE);
        	System.exit(0);
        }
        if (representitive_timestep<1 || representitive_timestep>num_timesteps) {
        	JOptionPane.showMessageDialog(null, "Representitive slice and timestep must be between 1 and the number of slices/timesteps. You selected "+representitive_timestep
        			+ " and the maximum is " + num_slices, "Error", JOptionPane.ERROR_MESSAGE);
        	System.exit(0);
        }
        
        
        
        int[] channels_to_leave = new int[(int)num_channels_to_leave];
        boolean unmix;
        int i = 0;
        for (int c=0; c<num_channels; c++) {
        	unmix = false;
        	for(int c1=0; c1<channels_to_unmix.length; c1++) {
        		if (channels_to_unmix[c1] == c) {
        			unmix=true;
        		}
        	}
        	if (!unmix) {channels_to_leave[i]=c; i++;}
        }

        INDArray pixel_value_matrix = Nd4j.zeros(num_cols, num_rows, num_channels, num_slices, num_timesteps);
        INDArray pixel_value_array_all = Nd4j.zeros(num_cols*num_rows*num_slices*num_timesteps, num_channels_to_unmix);
        
        double pixel = 0; 
        int[] coord;
        double sum = 0;
        int x, y, z, c, t;
        x = y = z = c = t = 0;
        for ( final T type : image )
        {
        	coord = new int[]{x, y, c, z, t};
        	pixel = type.getRealDouble();
        	sum = sum+pixel;
        	pixel_value_matrix.putScalar(coord, pixel);
        	if (x < num_cols-1)          {x++;}
        	else if (y < num_rows-1)     {x=0; y++;}
        	else if (c < num_channels-1) {x=0; y=0; c++;}
        	else if (z < num_slices-1)   {x=0; y=0; c=0; z++;}
        	else if (t < num_timesteps-1){x=0; y=0; c=0; z=0; t++;}

        }

        pixel_value_matrix = pixel_value_matrix.div(pixel_value_matrix.amax().toIntVector()[0]).mul(Short.MAX_VALUE);
        
        //Put images into pixel array ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        int[] indices;
        int[] pixel_value_array_indices;
        int pixel_value;
        int pixel_num = 0;

        z = representitive_slice-1;
        t = representitive_timestep-1;        
        
        //put channels to unmix from matrix to array
        long new_shape[] = {num_rows*num_cols, num_channels};
        INDArray pixel_value_array = pixel_value_matrix.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(z), NDArrayIndex.point(t)).reshape(new_shape);
        pixel_value_array = pixel_value_array.getColumns(channels_to_unmix);
        
        
	    
        pixel_num=0;
	    for (t = 0; t < num_timesteps; t++) {
		  for (z = 0; z < num_slices; z++) {
		    for (x = 0; x < num_cols; x++) {
			  for (y = 0; y < num_rows; y++) {
			    for (i = 0; i < num_channels_to_unmix; i++) {
	                	c = channels_to_unmix[i];
						indices = new int[]{x, y, c, z, t};
						pixel_value = pixel_value_matrix.getInt(indices);
						pixel_value_array_indices = new int[]{pixel_num, i};
						pixel_value_array_all.putScalar(pixel_value_array_indices, pixel_value);
					}
	                pixel_num++;
				}
	        }
          }
	    }
	    
        
        
        //scale pixel array~~~~~~~~~~~~~~~~~~~~~~~o~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if (intensity_ratios) {
        	INDArray pixel_total_intensity_array = pixel_value_array.sum(1); 
        	INDArray scaled_pixel_value_array = pixel_value_array.divColumnVector(pixel_total_intensity_array);
        	pixel_value_array = scaled_pixel_value_array;
        	
        	pixel_total_intensity_array = pixel_value_array_all.sum(1);
        	scaled_pixel_value_array = pixel_value_array_all.divColumnVector(pixel_total_intensity_array);
        	pixel_value_array_all = scaled_pixel_value_array;
        } 
        
        
        //Normalize each channel with z-score~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        INDArray mu = pixel_value_array.mean(0);
        INDArray sigma = pixel_value_array.std(0);
        pixel_value_array = pixel_value_array.subRowVector(mu).divRowVector(sigma);
        pixel_value_array_all = pixel_value_array_all.subRowVector(mu).divRowVector(sigma);


        //Run k-means on pixel array~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
		double min_distance = Double.MAX_VALUE;
        int best_replicate=0;
        List<INDArray> cluster_centers_best = new ArrayList<INDArray>();
	    INDArray cluster_center;
	    INDArray distances = Nd4j.zeros(num_cols*num_rows, k);
	    INDArray channel_distances = Nd4j.zeros(num_cols, num_rows, 1);
        //INDArray pixel_cluster_distances = Nd4j.zeros(num_cols*num_rows*1*1, k);
        double total_distance;
        List<INDArray> pixel_value_list = new ArrayList<INDArray>();
        for (i=0; i<num_cols*num_rows; i++) {
        	pixel_value_list.add(pixel_value_array.getRow(i));//This should get the whole row all at once
        }
        List<INDArray> locations = pixel_value_list;
        List<LocationWrapper> clusterInput = new ArrayList<LocationWrapper>(locations.size());
        for (INDArray location : locations) {
            clusterInput.add(new LocationWrapper(location));
        }

        System.out.println("Running k-means clustering");
        int cluster_count;
        INDArray cluster_centroid;
        
        
        for(int replicate=0; replicate<num_replicates; replicate++) {
        	ArrayList<INDArray> cluster_centers= new ArrayList<INDArray>();
            System.out.println("\nReplicate #"+(replicate+1));
	        KMeansPlusPlusClusterer<LocationWrapper> clusterer = new KMeansPlusPlusClusterer<LocationWrapper>(k, max_iter);
	        //System.out.println("Defined k-means model");
	        List<CentroidCluster<LocationWrapper>> clusterResults = clusterer.cluster(clusterInput);
	        //System.out.println("Finished clustering this replicate");
			

			for (i=0; i<clusterResults.size(); i++) {
			    System.out.println("Cluster " + (i+1));
			    cluster_centroid = Nd4j.zeros(num_channels_to_unmix);
			    cluster_count=0;
			    for (LocationWrapper locationWrapper : clusterResults.get(i).getPoints()) {
			    	cluster_centroid = cluster_centroid.add(locationWrapper.getLocation());
			    	cluster_count++;
			    }
			    cluster_centroid=cluster_centroid.div(cluster_count);
			    cluster_centers.add(cluster_centroid);
			    System.out.println(cluster_centroid.toString());
			}
	    	
	    	//Calculating total sum of distances from cluster centroids

	    	total_distance = 0;
	    			
        	for (c=0; c<k; c++) {
        		cluster_center = cluster_centers.get(c);
        		channel_distances = pixel_value_array.subRowVector(cluster_center);
        		channel_distances = channel_distances.norm2(1);
        		distances.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(c)}, channel_distances);
        	}
        	distances = distances.min(1);
        	total_distance = distances.meanNumber().doubleValue();
	    	
	        
	        //avg_distance = pixel_cluster_distances.max(1).amean(0).getDouble(0);
	        System.out.println("Average Distance from Cluster Centroid: "+total_distance/(num_rows*num_cols));
	        
	        if (total_distance <= min_distance) {
	        	min_distance = total_distance;
	        	best_replicate=replicate+1;
	        	cluster_centers_best=cluster_centers;
	        }
        }
        
        
		System.out.println("Using Replicate #"+(best_replicate));
		
		//Finding the background cluster
		double cluster_peak;
		INDArray cluster_peaks = Nd4j.zeros(k);
		for (i=0; i<k; i++) {
			cluster_center = cluster_centers_best.get(i);
			cluster_peak = -cluster_center.max(0).getDouble(0);
			cluster_peaks = cluster_peaks.putScalar(i, cluster_peak);
		}
		
		int background_cluster_index = cluster_peaks.getRow(0).argMax(1).getInt(0);
		System.out.print("Background cluster: ");
		System.out.println(cluster_centers_best.get(background_cluster_index));
		
		
		//Assigning a cluster to all pixels from original image
		INDArray pixel_cluster_distances_all = Nd4j.zeros(num_cols*num_rows*num_slices*num_timesteps, k);
	    INDArrayIndex channel_indices[] = {NDArrayIndex.all(), NDArrayIndex.point(0)};
				
		for (c=0; c<k; c++) {
			channel_indices[1] = NDArrayIndex.point(c);
			cluster_center = cluster_centers_best.get(c);
			distances = pixel_value_array_all.subRowVector(cluster_center);
			distances = distances.norm2(1);
			pixel_cluster_distances_all.put(channel_indices, distances);
		}
		
    	
        int[] pixel_cluster_array = pixel_cluster_distances_all.mul(-1).argMax(1).toIntVector(); 
        
        
        int[] cluster_counts = new int[k];
        for (i=0; i< pixel_cluster_array.length; i++) {
        	cluster_counts[pixel_cluster_array[i]]++;
        }
        
        
        for (i=0; i<cluster_counts.length;i++) {
        System.out.println("Cluster "+i+" has "+cluster_counts[i] + " pixels.");
        }
        
        int num_output_channels;
        INDArray complete_output_image = null;
        INDArray unmixed_channel;
        
        int[] pixel_cluster_matrix_indices;
        INDArray pixel_cluster_matrix = Nd4j.zeros(num_cols, num_rows, k, num_slices, num_timesteps);
        pixel_num=0;
        for (t = 0; t < num_timesteps; t++) {
	        for (z = 0; z < num_slices; z++) {
	        	for (x = 0; x < num_cols; x++) {
					for (y = 0; y < num_rows; y++) {
	                	c = pixel_cluster_array[pixel_num];
						pixel_cluster_matrix_indices = new int[]{x, y, c, z, t};
						pixel_cluster_matrix.putScalar(pixel_cluster_matrix_indices, 1);
		                pixel_num++;
					}
		        }
	        }
        }
        
        
		if (remove_background_only){
			num_output_channels = (int)num_channels;
			complete_output_image = Nd4j.zeros(num_cols, num_rows, num_output_channels, num_slices, num_timesteps);
			INDArray masked_unmixed_channel = Nd4j.zeros(num_cols, num_rows, num_slices, num_timesteps);
			INDArray mask = pixel_cluster_matrix.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(background_cluster_index), NDArrayIndex.all(), NDArrayIndex.all());
			long total_pixels = num_cols*num_rows*num_slices*num_timesteps;
			int allowed_pixels = mask.sum(0).sum(0).sum(0).sum(0).sum(0).getInt(0);
			System.out.println(allowed_pixels +" out of " + total_pixels);
			
			for (c=0; c<num_channels; c++) {
	        	unmixed_channel = pixel_value_matrix.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(c), NDArrayIndex.all(), NDArrayIndex.all());
	        	masked_unmixed_channel = masked_unmixed_channel.putWhereWithMask(mask, unmixed_channel);
	        	complete_output_image.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(c), NDArrayIndex.all(), NDArrayIndex.all()}, masked_unmixed_channel);
	        }
   
		}else {
	        //Use output classes to create image~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	        //Multiply outputs by input intensities~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	        INDArray pixel_cluster_intensity_matrix = pixel_cluster_matrix.mul(Short.MAX_VALUE);
	        boolean debug= false;
	        if(!debug) {
	        	INDArray pixel_value_matrix_channels_to_unmix = pixel_value_matrix.dup();
	        	for (c=0; c<channels_to_leave.length; c++) {
	        		pixel_value_matrix_channels_to_unmix.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(channels_to_leave[c]), NDArrayIndex.all(), NDArrayIndex.all()}, 0);
	        	}
		        INDArray pixel_total_intensity_matrix_compact = pixel_value_matrix_channels_to_unmix.max(2);
		        INDArray pixel_total_intensity_matrix_expanded = Nd4j.zeros(num_cols, num_rows, k, num_slices, num_timesteps);
		        for (int cluster = 0; cluster<k; cluster++) {
		            pixel_total_intensity_matrix_expanded.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(cluster), NDArrayIndex.all(), NDArrayIndex.all()}, pixel_total_intensity_matrix_compact);
		        }
		        pixel_cluster_intensity_matrix = pixel_cluster_matrix.mul(pixel_total_intensity_matrix_expanded);
	        }
	        
	        //Combine channels that are left alone with channels that come from clusters~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	        num_output_channels = k+(int)num_channels_to_leave;
	        complete_output_image = Nd4j.zeros(num_cols, num_rows, num_output_channels, num_slices, num_timesteps);
	        int channel;
	        for (c=0; c<num_channels_to_leave; c++) {
	        	channel = channels_to_leave[c];
	        	unmixed_channel = pixel_value_matrix.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(channel), NDArrayIndex.all(), NDArrayIndex.all());
	        	complete_output_image.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(channel), NDArrayIndex.all(), NDArrayIndex.all()}, unmixed_channel);
	        }
	        complete_output_image.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(num_channels_to_leave, num_output_channels), NDArrayIndex.all(), NDArrayIndex.all()}, pixel_cluster_intensity_matrix);
	        
		}
        //Output all channels into one image~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        System.out.println("Outputting Image");
        long[] dims = {num_cols, num_rows, num_output_channels, num_slices, num_timesteps};
        final String name = "Unmixed Image";
        final AxisType[] axes = { Axes.X, Axes.Y, Axes.CHANNEL, Axes.Z, Axes.TIME};
        dataset = datasetService.create(new UnsignedShortType(), dims, name, axes); //dataset = datasetService.create(dims, name, axes, bitsPerPixel, signed, floating);
        int plane_num = 0;
        //short test_pixel;
        short[] plane_to_add = new short[(int) (num_cols * num_rows)];//byte[] plane_to_add = new byte[(int) (num_cols * num_rows)];
        
        
        for (t = 0; t < num_timesteps; t++) {
	        for (z = 0; z < num_slices; z++) {
		        for (c = 0; c < num_output_channels; c++) {
		        	//System.out.println( x + " " + y +  " " + c + " " + z + " " + t);
					for (x = 0; x < num_cols; x++) {
						for (y = 0; y < num_rows; y++) {
							final int index = (int) (y * num_cols + x);
							indices = new int[]{x, y, c, z, t};
							plane_to_add[index] = (short)complete_output_image.getInt(indices);//(byte) complete_output_image.getDouble(indices);
						}
					}
					dataset.setPlane(plane_num, plane_to_add);
					plane_to_add = new short[(int) (num_cols * num_rows)];//new byte[(int) (num_cols * num_rows)];
					plane_num++;
		        }
	        }
        }
        System.out.println("Done");
    }
    
    public static class LocationWrapper implements Clusterable {
    	//http://commons.apache.org/proper/commons-math/userguide/ml.html
        private double[] points;
        private INDArray location;

        public LocationWrapper(INDArray location) {
            this.location = location;
            this.points = location.toDoubleVector();
        }

        public INDArray getLocation() {
            return location;
        }

        public double[] getPoint() {
            return points;
        }
    }

	public static void main(final String... args) throws Exception {
        // create the ImageJ application context with all available services
        final ImageJ ij = new ImageJ();
        ij.ui().showUI();

        
        // ask the user for a file to open
        final File file = ij.ui().chooseFile(null, "open");

        if (file != null) {
            // load the dataset
            final Dataset dataset = ij.scifio().datasetIO().open(file.getPath());

            // show the image
            ij.ui().show(dataset);

            // invoke the plugin
            ij.command().run(K_Means.class, true);
        }
    }

}

# simulate-and-recover
cogs 106 final assignment

Credit: I used ChatGBT to help me with writing and debugging all of the code in this project.

This assignment was designed to test if data designed by an EZ diffusion model could be recovered accurately using the estimation procedure.
My project uses the equations we were given in class to create an EZ diffusion model and then recovers it with the inverse equations to test this.

I created a class called EZDiffusion to simulate the data through the EZ diffusion model.
First, i selected "true parameters" for (ν,α,τ) and a sample size N. This is done in the generate_parameters function.
Then i generated the "predicted" summary statistics (Rpred,Mpred,Vpred) using the forward EZ equations. This is done in the compute_predicted_stats function. However, this doesn't account for the noise and chance that occurs in normal data collection, and I want to test the EZ diffusion model while acountiing for that. In order to make this data more "noisy", I used the equations covered in class to simulate observed statistics from the sampling distribution. This is done in the simulate_observations function.
Next, I generated the "estimated parameters" using the inverse EZ equations. This is done in the inverse_equations function. 
Finally, in order to test how good my EZ diffusion model was at recovering its own parameters, I had to calculate the estimation bias and the squared error. This is done in the compute_bias function.

My run_simulation function runs this process for 1000 iterations and collects the resulting biases and squared error values. From here, the data is passed into my main.py file where the calculate_average function computes the average bias and average squared error for ν, α, and τ. Then the write_to_csv function repeats the process for 3 different sample sizes (N = 10, 40, and 4000) and adds all the final results to a csv file (located in the src directory). 

My results show that on average the bias results are very close to 0 and that the squared error values always decrease as the sample size increases. This can be sen by looking at the results in the csv file or through several of the test cases that were written to double check it. I also wrote a test case that proved bias was exactly 0 when the observed data was equal to the predicted data. Because of these results, my model shows that data generated by an EZ diffusion model can in fact be accurately recovered by the estimation procedure. This is important because it is not something that all models can achieve and proves that the EZ diffusion model can be a very helpful tool in simulating and recovering data because of how reliable it is.

Note: I also have a function at the end of my EZDiffusion class called run_full_simulation. This was how I originally intended to run the full simulation with the three different sample sizes and collect all the results. This function collects all of the bias and squared error results (1 per iteration). However, while I was working I realized it was probably best to collect the averaged results instead because it made more sense when it came time to analyze and display the results. This is now done through the two functions in my main.py fle instead. The original run_full_simulation function is not called anywhere else within the code, but I stll kept the function up in case that was how we were supposed to collect the data.

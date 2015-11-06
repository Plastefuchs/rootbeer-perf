// TODO: cli parameter

package org.trifort.rootbeer.pi;

import org.trifort.rootbeer.runtime.Rootbeer;
import org.trifort.rootbeer.runtime.GpuDevice;
import org.trifort.rootbeer.runtime.Context;
import org.trifort.rootbeer.runtime.ThreadConfig;
import org.trifort.rootbeer.runtime.StatsRow;
import org.trifort.rootbeer.runtime.CacheConfig;
import java.util.List;
import java.util.Arrays;
import java.util.Random;
import java.util.ArrayList;
import java.io.FileWriter;
import java.io.File;
import java.io.IOException;

import java.util.Date;
import java.text.DateFormat;
import java.text.SimpleDateFormat;

import java.util.HashMap;
import java.util.Map;

public class GPUPi {
	// GPUPi extends Helper
	// size of the array
	private int threadCount;
	private int numberOfMultiProcessors; // 14
	private int blocksPerMultiProcessor; // 512
	private int numberOfRuns;
	private String fileName;
	private int numberOfIterationsPerKernel;
	private boolean outputConsoleStats = false;

	// ugly way to create a second file with the parameters of the program
	private Map parameter = new HashMap();

	GPUPi(int threadCount, int numberOfMultiProcessors,
			int blocksPerMultiProcessor, int numberOfRuns,
			int numberOfIterationsPerKernel) {
		this.threadCount = threadCount;
		this.numberOfMultiProcessors = numberOfMultiProcessors;
		this.blocksPerMultiProcessor = blocksPerMultiProcessor;
		this.numberOfRuns = numberOfRuns;
		this.numberOfIterationsPerKernel = numberOfIterationsPerKernel;

		DateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd_HHmmss");
		Date date = new Date();
		this.fileName = ("pi_" + dateFormat.format(date));

	}

	private void writeParameterFile() {
		try {
			File parameterFile = new File(this.fileName + "_parameter.csv");
			FileWriter writer = new FileWriter(parameterFile);
			for (Object key : this.parameter.keySet()) {
				System.out.println(key + " - " + parameter.get(key));
				writer.append(key + "," + parameter.get(key));
				writer.append("\n");
			}
			writer.append("Zeiten in Millisekunden.");
			writer.flush();
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void generateCsvFile(String sFileName, List<java.lang.Number> stats,
			List<String> header) {

		String COMMA_DELIMITER = ",";
		String NEW_LINE_SEPARATOR = "\n";

		try {
			File csvFile = new File(sFileName);
			// if no file exists, create a proper header
			if (!csvFile.exists()) {
				FileWriter writer = new FileWriter(csvFile, true);
				for (String item : header) {
					writer.append(item);
					writer.append(COMMA_DELIMITER);
				}
				writer.append(NEW_LINE_SEPARATOR);
				writer.flush();
				writer.close();
			}
			// write the stats to the csv file
			FileWriter writer = new FileWriter(csvFile, true);

			for (java.lang.Number item : stats) {
				writer.append(item.toString());
				writer.append(COMMA_DELIMITER);
			}
			writer.append(NEW_LINE_SEPARATOR);
			writer.flush();
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public double cpuPi(long tries) {
		double inCircle = 0;
		LinearCongruentialRandomGenerator lcg = new LinearCongruentialRandomGenerator(
				System.nanoTime());
		for (int i = 0; i < tries; i++) {
			double x = lcg.nextDouble();
			double y = lcg.nextDouble();
			double dist = Math.sqrt(x * x + y * y);
			if (dist <= 1.0) {
				inCircle++;
			}

		}
		double pi = 4 * inCircle / tries;
		return pi;

	}

	public void computePi() {

		// should have 192 threads per SM

		// int sizeBy2 = this.threadCount / 2;
		int sizeBy2 = this.threadCount;
		
		int outerCount = this.numberOfMultiProcessors
				* this.blocksPerMultiProcessor;
		// create array to hold each thread's hit number
		long[] array = new long[threadCount];

		Rootbeer rootbeer = new Rootbeer();
		List<GpuDevice> devices = rootbeer.getDevices();
		GpuDevice device0 = devices.get(0);
		Context context0 = device0.createContext();
		context0.setCacheConfig(CacheConfig.PREFER_SHARED);
		// set threadCountX, blockCountX, threadNumber
		// context0.setThreadConfig(this.threadCount, outerCount, outerCount *
		// this.threadCount);
		context0.setThreadConfig(this.threadCount, 1, this.threadCount);

		// context0.setThreadConfig(this.threadCount, outerCount, outerCount *
		// this.threadCount);
		// context0.setThreadConfig(sizeBy2, outerCount, outerCount *
		// this.threadCount);

		context0.setKernel(new GPUPiKernel(System.nanoTime(), array,
				this.numberOfIterationsPerKernel));
		context0.buildState();

		// Lists to create average stats
		List<Long> driverMemcopyToDeviceTimeList = new ArrayList<Long>(1);
		List<Long> driverExecTimeList = new ArrayList<Long>(1);
		List<Long> driverMemcopeFromDeviceTimeList = new ArrayList<Long>(1);
		List<Long> totalDriverExecTimeList = new ArrayList<Long>(1);
		List<Long> deserialTimeList = new ArrayList<Long>(1);
		List<Long> gpuRequiredMemList = new ArrayList<Long>(1);
		List<Long> gpuTimeList = new ArrayList<Long>(1);
		List<Double> ratioList = new ArrayList<Double>(1);

		int runs = 0;

		// limit the run
		while (runs < this.numberOfRuns) {
			runs += 1;

			LinearCongruentialRandomGenerator lcg = new LinearCongruentialRandomGenerator(
					System.nanoTime() / 2);

			// double cpuPi = this.cpuPi(1000000);
			// System.out.println(cpuPi);
			// start stopwatch
			long gpuStart = System.currentTimeMillis();
			// run the cached throughput mode state.
			// the data now reachable from the only
			// GPUPiKernel is serialized to the GPU
			context0.run();

			// stats and stat output
			long gpuStop = System.currentTimeMillis();
			long gpuTime = gpuStop - gpuStart;

			// System.out.println(Arrays.toString(array));

			// compute pi from the array
			long sum = 0;
			for (int i = 0; i < sizeBy2; ++i) {
				sum += array[i];
			}
			int tries = sizeBy2 * this.numberOfIterationsPerKernel;
			System.out.println("GPU Tries: " + tries);
			double pi = sum * 4 / (double) (tries);
			System.out.println(pi);
			long cpuStart = System.currentTimeMillis();

			double cpuPi = cpuPi(tries);

			System.out.println("CPU Tries: " + tries);
			System.out.println(cpuPi);
			long cpuStop = System.currentTimeMillis();
			long cpuTime = cpuStop - cpuStart;

			// pull stats from the context
			StatsRow row0 = context0.getStats();

			if (this.outputConsoleStats) {
				System.out
						.println("The serialization time of each first run is an anomaliy and should either be looked into further or discarded");
				System.out.println("serialization_time: "
						+ row0.getSerializationTime());
				System.out.println("driver_memcopy_to_device_time: "
						+ row0.getDriverMemcopyToDeviceTime());
				System.out.println("driver_execution_time: "
						+ row0.getDriverExecTime());
				System.out.println("driver_memcopy_from_device_time: "
						+ row0.getDriverMemcopyFromDeviceTime());
				System.out.println("total_driver_execution_time: "
						+ row0.getTotalDriverExecutionTime());
				System.out.println("deserialization_time: "
						+ row0.getDeserializationTime());
				System.out.println("gpu_required_memory: "
						+ context0.getRequiredMemory());
				System.out.println("gpu_time: " + gpuTime);
				System.out.println("cpu_time: " + cpuTime);
			}

			// stat array for csv file
			List<java.lang.Number> stats = new ArrayList<java.lang.Number>(1);
			List<String> statsHeader = new ArrayList<String>(1);

			// csv header
			statsHeader.add("serialization_time");
			statsHeader.add("driver_memcopy_to_device_time");
			statsHeader.add("driver_execution_time");
			statsHeader.add("driver_memcopy_from_device_time");
			statsHeader.add("total_driver_execution_time");
			statsHeader.add("deserialization_time");
			statsHeader.add("gpu_required_memory");
			statsHeader.add("gpu_time");
			statsHeader.add("cpu_time");

			statsHeader.add("Thread Count");
			statsHeader.add("Number of Multi Processors");
			statsHeader.add("Blocks per Multiprocessor");
			statsHeader.add("Number of Runs");
			statsHeader.add("Iterations per Kernel");
			
			// csv stats
			// the serialization time of the first run is higher than any of the
			// later runs. there is no clear indication why that is the case
			stats.add(row0.getSerializationTime());
			stats.add(row0.getDriverMemcopyToDeviceTime());
			stats.add(row0.getDriverExecTime());
			stats.add(row0.getDriverMemcopyFromDeviceTime());
			stats.add(row0.getTotalDriverExecutionTime());
			stats.add(row0.getDeserializationTime());
			stats.add(context0.getRequiredMemory());
			stats.add(gpuTime);
			stats.add(cpuTime);
			
			stats.add(this.threadCount);
			stats.add(this.numberOfMultiProcessors);
			stats.add(this.blocksPerMultiProcessor);
			stats.add(this.numberOfRuns);
			stats.add(this.numberOfIterationsPerKernel);

			String fileNameInstance = this.fileName + ".csv";
			generateCsvFile(fileNameInstance, stats, statsHeader);

		}

		// System.out.println("Finished " + runs + " pi runs.");

		// context0.close();

	}

	public static void main(String[] args) {
		// size of the inner arrays

		int threadCount = 1024; // 2048

		// number of processors and block size defines the number of inner
		// arrays
		// numMultiProcessors*blocksPerMultiProcessor;
		int numberOfMultiProcessors = 2; // 14
		int blocksPerMultiProcessor = 512; // 512
		int numberOfRuns = 1;
		int numberOfIterationsPerKernel = 1000000;

		int argThreadCount = Integer.parseInt(args[0]);
		int argNumbersOfMultiProcessors = Integer.parseInt(args[1]);
		int argBlocksPerMultiProcessor = Integer.parseInt(args[2]);
		int argNumberOfRuns = Integer.parseInt(args[3]);
		int argNumberOfIterationsPerKernel = Integer.parseInt(args[4]);

		if (0 != argThreadCount) {
			threadCount = argThreadCount;
		}
		if (0 != argNumbersOfMultiProcessors) {
			numberOfMultiProcessors = argNumbersOfMultiProcessors;
		}
		if (0 != argBlocksPerMultiProcessor) {
			blocksPerMultiProcessor = argBlocksPerMultiProcessor;
		}
		if (0 != argNumberOfRuns) {
			numberOfRuns = argNumberOfRuns;
		}
		if (0 != argNumberOfIterationsPerKernel) {
			numberOfIterationsPerKernel = argNumberOfIterationsPerKernel;
		}

		GPUPi sorter = new GPUPi(threadCount, numberOfMultiProcessors,
				blocksPerMultiProcessor, numberOfRuns,
				numberOfIterationsPerKernel);
		sorter.computePi();
	}
}

// Titan>
// http://www.anandtech.com/show/6760/nvidias-geforce-gtx-titan-part-1/4
// 2888 Cores, 6143 MB Dedicated Memory
// 32 threads / warp | 64 warps / SM
// 2048 threads / SM
// 255kb register per thread
// 64 FP64 CUDA cores -> 2 FP64 SM
// 14 SM
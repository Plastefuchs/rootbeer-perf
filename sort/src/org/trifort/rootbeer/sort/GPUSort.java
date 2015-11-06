// TODO: add cli parameter
// TODO: refactor to have methods for context creation

package org.trifort.rootbeer.sort;

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

public class GPUSort {

	// size of the inner arrays
	private int arraySize = 16; // 2048

	// number of processors and block size defines the number of inner arrays
	// numMultiProcessors*blocksPerMultiProcessor;
	private int numberOfMultiProcessors = 2; // 14
	private int blocksPerMultiProcessor = 256; // 512
	private int numberOfRuns = 10;
	private String fileName;
	
	private boolean outputConsoleStats = true;
	
	// ugly way to create a second file with the parameters of the program

	private Map parameter = new HashMap();

	GPUSort(int arraySize, int numberOfMultiProcessors,
			int blocksPerMultiProcessor, int numberOfRuns) {
		this.arraySize = arraySize;
		this.numberOfMultiProcessors = numberOfMultiProcessors;
		this.blocksPerMultiProcessor = blocksPerMultiProcessor;
		this.numberOfRuns = numberOfRuns;

		DateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd_HHmmss");
		Date date = new Date();
		this.fileName = ("sort_" + dateFormat.format(date));
		
	}

	private void writeParameterFile(){
		try{
		File parameterFile = new File(this.fileName + "_parameter.csv");
		FileWriter writer = new FileWriter(parameterFile);
		for (Object key: this.parameter.keySet()){
			System.out.println(key + " - " + parameter.get(key));
			writer.append(key + " - " + parameter.get(key));
			writer.append("\n");
		}
		writer.flush();
		writer.close();
		}
		catch (IOException e){
			e.printStackTrace();
		}
	}
	
	private int[] newArray(int size) {
		int[] ret = new int[size];

		for (int i = 0; i < size; ++i) {
			ret[i] = i;
		}
		return ret;
	}



	public void checkSorted(int[] array, int outerIndex) {
		for (int index = 0; index < array.length; ++index) {
			if (array[index] != index) {
				for (int index2 = 0; index2 < array.length; ++index2) {
					System.out.println("array[" + index2 + "]: "
							+ array[index2]);
				}
				throw new RuntimeException("not sorted: " + outerIndex);
			}
		}
	}

	public void fisherYates(int[] array) {
		Random random = new Random();
		for (int i = array.length - 1; i > 0; i--) {

			int index = random.nextInt(i + 1); // one random index
			/*
			 * Returns a pseudorandom, uniformly distributed int value between 0
			 * (inclusive) and the specified value (exclusive), drawn from this
			 * random number generator's sequence.
			 */
			int a = array[index]; // content of the field
			array[index] = array[i]; // set content of the field to content of
			// the current i field
			array[i] = a; // set content of the current i field to the old
			// content of the field
		}
	}

	public double averageStat(List<? extends java.lang.Number> array,
			int numberOfRuns) {
		double allStats = 0;
		for (java.lang.Number item : array) {
			allStats += item.doubleValue();
		}
		return allStats / numberOfRuns;
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

	// Titan>
	// http://www.anandtech.com/show/6760/nvidias-geforce-gtx-titan-part-1/4
	// 2888 Cores, 6143 MB Dedicated Memory
	// 32 threads / warp | 64 warps / SM
	// 2048 threads / SM
	// 255kb register per thread
	// 64 FP64 CUDA cores -> 2 FP64 SM
	// 14 SM

	public void sort() {

		// should have 192 threads per SM

		int sizeBy2 = this.arraySize / 2;

		// set size of the outer array
		int outerCount = this.numberOfMultiProcessors
				* this.blocksPerMultiProcessor;

		// create array as wide as the number of SM * the blocks per SM
		int[][] array = new int[outerCount][];
		// create one size wide inner array per block(thread group)
		for (int i = 0; i < outerCount; ++i) {
			array[i] = newArray(this.arraySize);
		}

		Rootbeer rootbeer = new Rootbeer();
		List<GpuDevice> devices = rootbeer.getDevices();
		GpuDevice device0 = devices.get(0);
		// create a context
		// after you run you can call context0.getRequiredMemory() to see
		// what value to enter here
		Context context0 = device0.createContext(); // 4212880 as example
		// use more die area for shared memory instead of
		// cache. the shared memory is a software defined
		// cache that, if programmed properly, can perform
		// better than the hardware cache
		// see (CUDA Occupancy
		// calculator)[http://developer.download.nvidia.com/compute/cuda/CUDA_Occupancy_calculator.xls]
		context0.setCacheConfig(CacheConfig.PREFER_SHARED);
		// wire thread config for throughput mode. after
		// calling buildState, the book-keeping information
		// will be cached in the JNI driver
		context0.setThreadConfig(sizeBy2, outerCount, outerCount * sizeBy2);
		context0.setKernel(new GPUSortKernel(array));
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

			// randomize the array to be sorted
			for (int i = 0; i < outerCount; ++i) {
				fisherYates(array[i]);
			}

			// start stopwatch
			long gpuStart = System.currentTimeMillis();
			// run the cached throughput mode state.
			// the data now reachable from the only
			// GPUSortKernel is serialized to the GPU
			context0.run();
			// stats and stat output
			long gpuStop = System.currentTimeMillis();
			long gpuTime = gpuStop - gpuStart;

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
			
			statsHeader.add("Array Size/Thread Count");
			statsHeader.add("Full array length");
			statsHeader.add("Number of Multi Processors");
			statsHeader.add("Blocks per Multiprocessor");
			statsHeader.add("Number of Runs");

			// csv stats
			// the serialization time of the first run is higher than any of the
			// later runs
			// there is no clear indication why that is the case
			stats.add(row0.getSerializationTime());
			stats.add(row0.getDriverMemcopyToDeviceTime());
			stats.add(row0.getDriverExecTime());
			stats.add(row0.getDriverMemcopyFromDeviceTime());
			stats.add(row0.getTotalDriverExecutionTime());
			stats.add(row0.getDeserializationTime());
			stats.add(context0.getRequiredMemory());
			stats.add(gpuTime);
			
			stats.add(this.arraySize);
			stats.add(this.arraySize * outerCount);
			stats.add(this.numberOfMultiProcessors);
			stats.add(this.blocksPerMultiProcessor);
			stats.add(this.numberOfRuns);

			String fileNameInstance = this.fileName + ".csv";
			generateCsvFile(fileNameInstance, stats, statsHeader);

			// check that each array was properly sorted
			// shuffle afterwards for the CPU sorting
			for (int i = 0; i < outerCount; ++i) {
				checkSorted(array[i], i);
				fisherYates(array[i]);
			}
			// sort on CPU as reference for comparison
			long cpuStart = System.currentTimeMillis();
			for (int i = 0; i < outerCount; ++i) {
				Arrays.sort(array[i]);
			}
			long cpuStop = System.currentTimeMillis();
			long cpuTime = cpuStop - cpuStart;

			double ratio = (double) cpuTime / (double) gpuTime;
			if (this.outputConsoleStats) {
				System.out.println("cpu_time: " + cpuTime);
				System.out.println("ratio: " + ratio);
			}
			ratioList.add(ratio);

		}

		System.out.println("Finished " + runs + " sorting runs.");

		// context0.close();

	}

	public static void main(String[] args) {
		// size of the inner arrays
		int arraySize = 2048; // 2048

		// number of processors and block size defines the number of inner
		// arrays
		// numMultiProcessors*blocksPerMultiProcessor;
		int numberOfMultiProcessors = 14; // 14
		int blocksPerMultiProcessor = 512; // 512
		int numberOfRuns = 10;

		int argArraySize = Integer.parseInt(args[0]);
		int argNumbersOfMultiProcessors = Integer.parseInt(args[1]);
		int argBlocksPerMultiProcessor = Integer.parseInt(args[2]);
		int argNumberOfRuns = Integer.parseInt(args[3]);
		
		
		
		if (0 != argArraySize) {
			arraySize = argArraySize;
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
		
		GPUSort sorter = new GPUSort(arraySize, numberOfMultiProcessors,
				blocksPerMultiProcessor, numberOfRuns);
		sorter.sort();
		
	
		
	}
}

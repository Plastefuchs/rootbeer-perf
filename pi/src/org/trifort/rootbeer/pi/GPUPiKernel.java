package org.trifort.rootbeer.pi;

import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.RootbeerGpu;
import java.util.Arrays;
import java.util.Random;
import java.lang.Math;

public class GPUPiKernel implements Kernel {

	private long seed;
	private long[] resultArray;
	private int iterations; // iterations per kernel

	public GPUPiKernel(long seed, long[] resultArray, int iterations) {
		this.seed = seed;
		this.resultArray = resultArray;
		this.iterations = iterations;

	}

	@Override
	public void gpuMethod() {
		int blockId = RootbeerGpu.getBlockIdxx();
		int threadId = RootbeerGpu.getThreadIdxx();
		int globalThreadId = RootbeerGpu.getThreadIdxx()
				+ RootbeerGpu.getBlockIdxx() * RootbeerGpu.getBlockDimx();
		
		LinearCongruentialRandomGenerator lcg = new LinearCongruentialRandomGenerator(
				this.seed / globalThreadId);

		long hits = 0;
//		System.out.println(lcg.nextDouble());
		 
		for (int i = 0; i < iterations; i++) {

			// use lcg instead of math random, due to 702 crashes
			double x = lcg.nextDouble(); // value between 0 and 1
			double y = lcg.nextDouble(); // value between 0 and 1
//			double x = Math.random();
//			double y = Math.random();
			double erg = x * x + y * y;
			if (erg <= 1.0) {
				hits++;
			}
		}

		 resultArray[threadId] = hits;

		RootbeerGpu.syncthreads();
	}
}
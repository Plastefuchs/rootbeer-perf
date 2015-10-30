package org.trifort.rootbeer.sort;

import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.RootbeerGpu;
import java.util.Arrays;

public class GPUSortKernel implements Kernel {

  private int[][] arrays;

  public GPUSortKernel(int[][] arrays){
    this.arrays = arrays;
  }

  @Override
  public void gpuMethod(){
	  // one one-dimensional inner array per block
    int[] array = arrays[RootbeerGpu.getBlockIdxx()];
    int index1a = RootbeerGpu.getThreadIdxx() << 1;
    int index1b = index1a + 1;
    int index2a = index1a - 1;
    int index2b = index1a;

    // pick adress in the shared memory
    int index1a_shared = index1a << 2;
    int index1b_shared = index1b << 2;
    int index2a_shared = index2a << 2;
    int index2b_shared = index2b << 2;

    // set integer in shared memory. requires 4 bytes.
    // index is byte offset into shared memory
    // http://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/
    RootbeerGpu.setSharedInteger(index1a_shared, array[index1a]);
    RootbeerGpu.setSharedInteger(index1b_shared, array[index1b]);

    //outer pass
    int arrayLength = array.length >> 1;
    for(int i = 0; i < arrayLength; ++i){
      
      // read value from shared memory
      int value1 = RootbeerGpu.getSharedInteger(index1a_shared);
      int value2 = RootbeerGpu.getSharedInteger(index1b_shared);

//      debug output      
//      if (RootbeerGpu.getBlockIdxx() == 0){
//        	if ((RootbeerGpu.getThreadIdxx() << 1) == 0){
//              System.out.println(RootbeerGpu.getThreadIdxx()+" val1: "+value1+" val2: "+value2);
//        	}
      RootbeerGpu.syncthreads();
      int shared_value = value1;
    
      // compare & swap from right to left
      if(value2 < value1){
    	// save value to the right
        shared_value = value2;
        // swap the values
        RootbeerGpu.setSharedInteger(index1a_shared, value2);
        RootbeerGpu.setSharedInteger(index1b_shared, value1);
      }
      // wait for all threads
      RootbeerGpu.syncthreads();
      // we are not to the most left, 
      if(index2a >= 0){
        value1 = RootbeerGpu.getSharedInteger(index2a_shared);
        //value2 = RootbeerGpu.getSharedInteger(index2b_shared);
        // ?? why?
        value2 = shared_value;
        if(value2 < value1){
          RootbeerGpu.setSharedInteger(index2a_shared, value2);
          RootbeerGpu.setSharedInteger(index2b_shared, value1);
        }
      }
      RootbeerGpu.syncthreads();
    }
    // write both values back into the array
    array[index1a] = RootbeerGpu.getSharedInteger(index1a_shared);
    array[index1b] = RootbeerGpu.getSharedInteger(index1b_shared);
  }
}

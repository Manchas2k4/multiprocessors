// =================================================================
//	
// File: MergeSort.java
// Author: Pedro Perez
// Description: This file implements the non parallelr merge sort 
//				algorithm. 
//
// Copyright (c) 2023 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

import java.util.Arrays;

public class MergeSort {
	private int A[], B[], start, end;

	public MergeSort(int start, int end, int A[], int B[]) {
		this.start = start;
		this.end = end;
		this.A = A;
		this.B = B;
	}

	private void swap(int a[], int i, int j) {
		int aux = a[i];
		a[i] = a[j];
		a[j] = aux;
	}

	private void copyArray(int low, int high) {
		int length = high - low + 1;
		System.arraycopy(B, low, A, low, length);
	}

	private void merge(int low, int mid, int high) {
		int i, j, k;

		i = low;
		j = mid + 1;
		k = low;
		while(i <= mid && j <= high){
			if(A[i] < A[j]){
				B[k] = A[i];
				i++;
			}else{
				B[k] = A[j];
				j++;
			}
			k++;
		}
		for(; j <= high; j++){
			B[k++] = A[j];
		}

		for(; i <= mid; i++){
			B[k++] = A[i];
		}
	}

	private void split(int low, int high) {
		int  mid;

		if ((high - low + 1) == 1) {
			return;
		}

		mid = low + ((high - low) / 2);
		split(low, mid);
		split(mid + 1, high);
		merge(low, mid, high);
		copyArray(low, high);
	}

	public void doTask() {
		split(start, end);
	}
}
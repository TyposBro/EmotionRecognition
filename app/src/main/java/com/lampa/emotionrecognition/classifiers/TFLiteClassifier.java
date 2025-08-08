package com.lampa.emotionrecognition.classifiers;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.util.Log; // Make sure this import is present

import com.lampa.emotionrecognition.classifiers.behaviors.ClassifyBehavior;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

// Abstract classifier using tflite format
public abstract class TFLiteClassifier {
    protected AssetManager mAssetManager;

    protected Interpreter mInterpreter;

    protected Interpreter.Options mTFLiteInterpreterOptions;

    protected List<String> mLabels;

    protected ClassifyBehavior classifyBehavior;

    // THIS IS THE CORRECTED CONSTRUCTOR
    public TFLiteClassifier(AssetManager assetManager, String modelFileName, String[] labels) {
        mAssetManager = assetManager;
        mTFLiteInterpreterOptions = new Interpreter.Options();

        // --- START: MODERN GPU DELEGATE INITIALIZATION ---
        try {
            // Create a new options instance for the GpuDelegate.
            GpuDelegate.Options delegateOptions = new GpuDelegate.Options();
            GpuDelegate delegate = new GpuDelegate();
            mTFLiteInterpreterOptions.addDelegate(delegate);
            Log.d("TFLiteClassifier", "GPU Delegate created successfully.");
        } catch (Exception e) {
            // If the GPU delegate fails, log the error and fall back to using the CPU.
            // This prevents the app from crashing on unsupported devices.
            Log.e("TFLiteClassifier", "Failed to create GPU delegate. Using CPU instead.", e);
            mTFLiteInterpreterOptions.setNumThreads(4); // Optional: improve CPU performance
        }
        // --- END: MODERN GPU DELEGATE INITIALIZATION ---

        // Now, initialize the interpreter with the options (which may or may not have the GPU delegate)
        try {
            mInterpreter = new Interpreter(loadModel(modelFileName), mTFLiteInterpreterOptions);
        } catch (Exception ex) {
            Log.e("TFLiteClassifier", "Failed to initialize interpreter.", ex);
            ex.printStackTrace();
        }

        mLabels = new ArrayList<>(Arrays.asList(labels));
    }

    public MappedByteBuffer loadModel(String modelFileName) throws IOException {
        AssetFileDescriptor fileDescriptor = mAssetManager.openFd(modelFileName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());

        FileChannel fileChannel = inputStream.getChannel();

        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // Close the interpreter to avoid memory leaks
    public void close() {
        if (mInterpreter != null) {
            mInterpreter.close();
        }
    }

    public Interpreter getInterpreter() {
        return mInterpreter;
    }

    public List<String> getLabels() {
        return mLabels;
    }

    public void setLabels(List<String> labels) {
        mLabels = mLabels;
    }
}
package com.lampa.emotionrecognition.classifiers;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.util.Log;

import com.lampa.emotionrecognition.classifiers.behaviors.ClassifyBehavior;

import org.tensorflow.lite.Interpreter;

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

    public TFLiteClassifier(AssetManager assetManager, String modelFileName, String[] labels) {
        mAssetManager = assetManager;
        mTFLiteInterpreterOptions = new Interpreter.Options();

        // Use CPU with multiple threads for better performance
        // Avoiding GPU delegate due to compatibility issues
        mTFLiteInterpreterOptions.setNumThreads(4);
        Log.d("TFLiteClassifier", "Using CPU with 4 threads for TensorFlow Lite inference.");

        // Initialize the interpreter with the options
        try {
            mInterpreter = new Interpreter(loadModel(modelFileName), mTFLiteInterpreterOptions);
            Log.d("TFLiteClassifier", "Interpreter initialized successfully with CPU.");
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
            mInterpreter = null;
        }
    }

    public Interpreter getInterpreter() {
        return mInterpreter;
    }

    public List<String> getLabels() {
        return mLabels;
    }

    public void setLabels(List<String> labels) {
        mLabels = labels;
    }
}
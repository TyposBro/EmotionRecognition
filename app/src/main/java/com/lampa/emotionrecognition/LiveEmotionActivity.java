package com.lampa.emotionrecognition;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.YuvImage;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.lampa.emotionrecognition.classifiers.TFLiteImageClassifier;
import com.lampa.emotionrecognition.utils.ImageUtils;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class LiveEmotionActivity extends AppCompatActivity {
    private static final String TAG = "LiveEmotionActivity";
    private static final int CAMERA_PERMISSION_REQUEST_CODE = 200;

    private TextureView mPreviewView;
    private TextView mEmotionTextView;
    private Button mBackButton;
    private Button mSwitchCameraButton;

    private ProcessCameraProvider mCameraProvider;
    private Preview mPreview;
    private ImageAnalysis mImageAnalysis;
    private FaceDetector mFaceDetector;
    private TFLiteImageClassifier mEmotionClassifier;

    private boolean mIsProcessing = false;
    private boolean mIsFrontCamera = true;

    private Executor mCameraExecutor;
    private Handler mMainHandler;

    // For drawing overlay
    private Canvas mOverlayCanvas;
    private Bitmap mOverlayBitmap;
    private Paint mPaint;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_live_emotion);

        initializeViews();
        initializeCamera();
        initializeML();

        if (checkCameraPermission()) {
            startCamera();
        } else {
            requestCameraPermission();
        }
    }

    private void initializeViews() {
        mPreviewView = findViewById(R.id.camera_preview);
        mEmotionTextView = findViewById(R.id.emotion_text_view);
        mBackButton = findViewById(R.id.back_button);
        mSwitchCameraButton = findViewById(R.id.switch_camera_button);

        mBackButton.setOnClickListener(v -> finish());
        mSwitchCameraButton.setOnClickListener(v -> switchCamera());

        // Initialize paint for drawing
        mPaint = new Paint();
        mPaint.setColor(Color.GREEN);
        mPaint.setStrokeWidth(4f);
        mPaint.setTextSize(48f);
        mPaint.setAntiAlias(true);
    }

    private void initializeCamera() {
        mCameraExecutor = Executors.newSingleThreadExecutor();
        mMainHandler = new Handler(getMainLooper());
    }

    private void initializeML() {
        // Initialize face detector
        FaceDetectorOptions options = new FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                .setMinFaceSize(0.15f)
                .build();

        mFaceDetector = FaceDetection.getClient(options);

        // Initialize emotion classifier
        mEmotionClassifier = new TFLiteImageClassifier(
                getAssets(),
                "simple_classifier.tflite",
                getResources().getStringArray(R.array.emotions)
        );
    }

    private boolean checkCameraPermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED;
    }

    private void requestCameraPermission() {
        ActivityCompat.requestPermissions(this,
                new String[]{Manifest.permission.CAMERA},
                CAMERA_PERMISSION_REQUEST_CODE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera();
            } else {
                Toast.makeText(this, "Camera permission is required for live emotion detection",
                        Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                mCameraProvider = cameraProviderFuture.get();
                bindCameraUseCases();
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Error starting camera", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindCameraUseCases() {
        if (mCameraProvider == null) return;

        // Unbind all use cases before rebinding
        mCameraProvider.unbindAll();

        // Camera selector
        CameraSelector cameraSelector = mIsFrontCamera ?
                CameraSelector.DEFAULT_FRONT_CAMERA : CameraSelector.DEFAULT_BACK_CAMERA;

        // Preview use case
        mPreview = new Preview.Builder()
                .setTargetResolution(new Size(640, 480))
                .build();

        mPreview.setSurfaceProvider(mPreviewView.getSurfaceProvider());

        // Image analysis use case
        mImageAnalysis = new ImageAnalysis.Builder()
                .setTargetResolution(new Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        mImageAnalysis.setAnalyzer(mCameraExecutor, new EmotionAnalyzer());

        try {
            mCameraProvider.bindToLifecycle(this, cameraSelector, mPreview, mImageAnalysis);
        } catch (Exception e) {
            Log.e(TAG, "Failed to bind camera use cases", e);
        }
    }

    private void switchCamera() {
        mIsFrontCamera = !mIsFrontCamera;
        bindCameraUseCases();
    }

    private class EmotionAnalyzer implements ImageAnalysis.Analyzer {
        @Override
        public void analyze(@NonNull ImageProxy image) {
            if (mIsProcessing) {
                image.close();
                return;
            }

            mIsProcessing = true;

            try {
                InputImage inputImage = InputImage.fromMediaImage(
                        image.getImage(),
                        image.getImageInfo().getRotationDegrees()
                );

                detectFacesAndEmotions(inputImage);

            } catch (Exception e) {
                Log.e(TAG, "Error analyzing image", e);
                mIsProcessing = false;
            } finally {
                image.close();
            }
        }
    }

    private void detectFacesAndEmotions(InputImage inputImage) {
        Task<List<Face>> result = mFaceDetector.process(inputImage)
                .addOnSuccessListener(new OnSuccessListener<List<Face>>() {
                    @Override
                    public void onSuccess(List<Face> faces) {
                        processFaces(faces, inputImage);
                        mIsProcessing = false;
                    }
                })
                .addOnFailureListener(new OnFailureListener() {
                    @Override
                    public void onFailure(@NonNull Exception e) {
                        Log.e(TAG, "Face detection failed", e);
                        mIsProcessing = false;
                    }
                });
    }

    private void processFaces(List<Face> faces, InputImage inputImage) {
        if (faces.isEmpty()) {
            mMainHandler.post(() -> {
                mEmotionTextView.setText("No face detected");
                clearOverlay();
            });
            return;
        }

        StringBuilder emotionText = new StringBuilder();

        for (int i = 0; i < faces.size(); i++) {
            Face face = faces.get(i);
            Rect boundingBox = face.getBoundingBox();

            try {
                // Extract face bitmap
                Bitmap faceBitmap = extractFaceBitmap(inputImage, boundingBox);
                if (faceBitmap != null) {
                    // Classify emotion
                    Map<String, Float> emotions = mEmotionClassifier.classify(faceBitmap, true);
                    String dominantEmotion = getDominantEmotion(emotions);
                    float confidence = emotions.get(dominantEmotion) * 100;

                    emotionText.append(String.format("Face %d: %s (%.1f%%)",
                            i + 1, dominantEmotion, confidence));
                    if (i < faces.size() - 1) {
                        emotionText.append("\n");
                    }
                }
            } catch (Exception e) {
                Log.e(TAG, "Error processing face " + i, e);
            }
        }

        final String finalEmotionText = emotionText.toString();
        mMainHandler.post(() -> {
            mEmotionTextView.setText(finalEmotionText);
            drawFaceOverlays(faces);
        });
    }

    private Bitmap extractFaceBitmap(InputImage inputImage, Rect boundingBox) {
        try {
            // Convert InputImage to Bitmap
            Bitmap bitmap = inputImageToBitmap(inputImage);
            if (bitmap == null) return null;

            // Ensure bounding box is within bitmap bounds
            int left = Math.max(0, boundingBox.left);
            int top = Math.max(0, boundingBox.top);
            int right = Math.min(bitmap.getWidth(), boundingBox.right);
            int bottom = Math.min(bitmap.getHeight(), boundingBox.bottom);

            int width = right - left;
            int height = bottom - top;

            if (width <= 0 || height <= 0) return null;

            return Bitmap.createBitmap(bitmap, left, top, width, height);

        } catch (Exception e) {
            Log.e(TAG, "Error extracting face bitmap", e);
            return null;
        }
    }

    private Bitmap inputImageToBitmap(InputImage inputImage) {
        try {
            android.media.Image mediaImage = inputImage.getMediaImage();
            if (mediaImage == null) return null;

            // Get image dimensions
            int width = mediaImage.getWidth();
            int height = mediaImage.getHeight();

            // Get Y plane (luminance)
            android.media.Image.Plane yPlane = mediaImage.getPlanes()[0];
            ByteBuffer yBuffer = yPlane.getBuffer();
            byte[] yData = new byte[yBuffer.remaining()];
            yBuffer.get(yData);

            // Create grayscale bitmap from Y channel for faster processing
            int[] pixels = new int[width * height];
            for (int i = 0; i < yData.length; i++) {
                int gray = (yData[i] & 0xFF);
                pixels[i] = 0xFF000000 | (gray << 16) | (gray << 8) | gray;
            }

            Bitmap bitmap = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888);

            // Apply rotation if needed
            int rotationDegrees = inputImage.getRotationDegrees();
            if (rotationDegrees != 0) {
                Matrix matrix = new Matrix();
                matrix.postRotate(rotationDegrees);
                bitmap = Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, false);
            }

            return bitmap;

        } catch (Exception e) {
            Log.e(TAG, "Error converting InputImage to Bitmap", e);
            return null;
        }
    }

    private String getDominantEmotion(Map<String, Float> emotions) {
        String dominantEmotion = "";
        float maxConfidence = 0f;

        for (Map.Entry<String, Float> entry : emotions.entrySet()) {
            if (entry.getValue() > maxConfidence) {
                maxConfidence = entry.getValue();
                dominantEmotion = entry.getKey();
            }
        }

        return dominantEmotion;
    }

    private void drawFaceOverlays(List<Face> faces) {
        // This is a simplified overlay drawing - you might want to implement
        // a custom overlay view for better performance and accuracy
        // For now, we'll just update the text display
    }

    private void clearOverlay() {
        // Clear any overlay drawings
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        if (mCameraProvider != null) {
            mCameraProvider.unbindAll();
        }

        if (mEmotionClassifier != null) {
            mEmotionClassifier.close();
        }

        if (mCameraExecutor != null) {
            mCameraExecutor.shutdown();
        }
    }
}
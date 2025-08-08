package com.lampa.emotionrecognition;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Pair;
import android.view.View;
import android.widget.Button;
import android.widget.ExpandableListView;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Toast;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.lampa.emotionrecognition.classifiers.TFLiteImageClassifier;
import com.lampa.emotionrecognition.utils.ImageUtils;
import com.lampa.emotionrecognition.utils.SortingHelper;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class MainActivity extends AppCompatActivity {

    private static final int GALLERY_REQUEST_CODE = 0;
    private static final int TAKE_PHOTO_REQUEST_CODE = 1;
    private static final int CAMERA_PERMISSION_REQUEST_CODE = 100;
    private static final int STORAGE_PERMISSION_REQUEST_CODE = 101;

    private final String MODEL_FILE_NAME = "simple_classifier.tflite";
    private final int SCALED_IMAGE_BIGGEST_SIZE = 480;

    private TFLiteImageClassifier mClassifier;
    private ProgressBar mClassificationProgressBar;
    private ImageView mImageView;
    private Button mPickImageButton;
    private Button mTakePhotoButton;
    private Button mLiveEmotionButton;
    private ExpandableListView mClassificationExpandableListView;
    private Uri mCurrentPhotoUri;
    private Map<String, List<Pair<String, String>>> mClassificationResult;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mClassificationProgressBar = findViewById(R.id.classification_progress_bar);

        mClassifier = new TFLiteImageClassifier(
                this.getAssets(),
                MODEL_FILE_NAME,
                getResources().getStringArray(R.array.emotions));

        mClassificationResult = new LinkedHashMap<>();

        mImageView = findViewById(R.id.image_view);

        mPickImageButton = findViewById(R.id.pick_image_button);
        mPickImageButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                checkStoragePermissionAndPickImage();
            }
        });

        mTakePhotoButton = findViewById(R.id.take_photo_button);
        mTakePhotoButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                checkCameraPermissionAndTakePhoto();
            }
        });

        mLiveEmotionButton = findViewById(R.id.live_emotion_button);
        mLiveEmotionButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startLiveEmotionDetection();
            }
        });

        mClassificationExpandableListView = findViewById(R.id.classification_expandable_list_view);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        mClassifier.close();

        File picturesDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        if (picturesDir != null && picturesDir.listFiles() != null) {
            for (File tempFile : picturesDir.listFiles()) {
                tempFile.delete();
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == Activity.RESULT_OK) {
            switch (requestCode) {
                case GALLERY_REQUEST_CODE:
                    clearClassificationExpandableListView();
                    Uri pickedImageUri = data.getData();
                    processImageRequestResult(pickedImageUri);
                    break;
                case TAKE_PHOTO_REQUEST_CODE:
                    clearClassificationExpandableListView();
                    processImageRequestResult(mCurrentPhotoUri);
                    break;
                default:
                    break;
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        switch (requestCode) {
            case CAMERA_PERMISSION_REQUEST_CODE:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    takePhoto();
                } else {
                    Toast.makeText(this, "Camera permission is required to take photos", Toast.LENGTH_SHORT).show();
                }
                break;
            case STORAGE_PERMISSION_REQUEST_CODE:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    pickFromGallery();
                } else {
                    Toast.makeText(this, "Storage permission is required to access gallery", Toast.LENGTH_SHORT).show();
                }
                break;
        }
    }

    private void checkCameraPermissionAndTakePhoto() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
        } else {
            takePhoto();
        }
    }

    private void checkStoragePermissionAndPickImage() {
        // For Android 13+ (API 33+), we don't need READ_EXTERNAL_STORAGE for gallery access
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            pickFromGallery();
        } else if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, STORAGE_PERMISSION_REQUEST_CODE);
        } else {
            pickFromGallery();
        }
    }

    private void clearClassificationExpandableListView() {
        Map<String, List<Pair<String, String>>> emptyMap = new LinkedHashMap<>();
        ClassificationExpandableListAdapter adapter = new ClassificationExpandableListAdapter(emptyMap);
        mClassificationExpandableListView.setAdapter(adapter);
    }

    private void processImageRequestResult(Uri resultImageUri) {
        if (resultImageUri == null) {
            Toast.makeText(this, "Failed to get image.", Toast.LENGTH_SHORT).show();
            return;
        }
        Bitmap scaledResultImageBitmap = getScaledImageBitmap(resultImageUri);

        mImageView.setImageBitmap(scaledResultImageBitmap);

        mClassificationResult.clear();
        setCalculationStatusUI(true);
        detectFaces(scaledResultImageBitmap);
    }

    private void pickFromGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        intent.setType("image/*");
        startActivityForResult(intent, GALLERY_REQUEST_CODE);
    }

    private void takePhoto() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (intent.resolveActivity(getPackageManager()) != null) {
            File photoFile = null;
            try {
                photoFile = createImageFile();
            } catch (IOException e) {
                e.printStackTrace();
                Toast.makeText(this, "Error creating image file.", Toast.LENGTH_SHORT).show();
                return;
            }

            mCurrentPhotoUri = FileProvider.getUriForFile(
                    this,
                    "com.lampa.emotionrecognition.fileprovider",
                    photoFile);

            intent.putExtra(MediaStore.EXTRA_OUTPUT, mCurrentPhotoUri);
            startActivityForResult(intent, TAKE_PHOTO_REQUEST_CODE);
        }
    }

    private Bitmap getScaledImageBitmap(Uri imageUri) {
        Bitmap scaledImageBitmap = null;

        try {
            Bitmap imageBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);

            int scaledHeight;
            int scaledWidth;
            float scaleFactor;

            if (imageBitmap.getHeight() > imageBitmap.getWidth()) {
                scaledHeight = SCALED_IMAGE_BIGGEST_SIZE;
                scaleFactor = scaledHeight / (float) imageBitmap.getHeight();
                scaledWidth = (int) (imageBitmap.getWidth() * scaleFactor);
            } else {
                scaledWidth = SCALED_IMAGE_BIGGEST_SIZE;
                scaleFactor = scaledWidth / (float) imageBitmap.getWidth();
                scaledHeight = (int) (imageBitmap.getHeight() * scaleFactor);
            }

            scaledImageBitmap = Bitmap.createScaledBitmap(imageBitmap, scaledWidth, scaledHeight, true);
            scaledImageBitmap = ImageUtils.rotateToNormalOrientation(getContentResolver(), scaledImageBitmap, imageUri);

        } catch (IOException e) {
            e.printStackTrace();
        }

        return scaledImageBitmap;
    }

    private void detectFaces(Bitmap imageBitmap) {
        FaceDetectorOptions faceDetectorOptions = new FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                .setMinFaceSize(0.1f)
                .build();

        FaceDetector faceDetector = FaceDetection.getClient(faceDetectorOptions);
        final InputImage inputImage = InputImage.fromBitmap(imageBitmap, 0);

        Task<List<Face>> result = faceDetector.process(inputImage)
                .addOnSuccessListener(new OnSuccessListener<List<Face>>() {
                    @Override
                    public void onSuccess(List<Face> faces) {
                        Bitmap imageBitmap = inputImage.getBitmapInternal();
                        Bitmap tmpBitmap = Bitmap.createBitmap(imageBitmap.getWidth(), imageBitmap.getHeight(), imageBitmap.getConfig());

                        Canvas tmpCanvas = new Canvas(tmpBitmap);
                        tmpCanvas.drawBitmap(imageBitmap, 0, 0, null);

                        Paint paint = new Paint();
                        paint.setColor(Color.GREEN);
                        paint.setStrokeWidth(2);
                        paint.setTextSize(48);

                        final float textIndentFactor = 0.1f;

                        if (!faces.isEmpty()) {
                            int faceId = 1;

                            for (Face face : faces) {
                                Rect faceRect = getInnerRect(face.getBoundingBox(), imageBitmap.getWidth(), imageBitmap.getHeight());

                                paint.setStyle(Paint.Style.STROKE);
                                tmpCanvas.drawRect(faceRect, paint);

                                paint.setStyle(Paint.Style.FILL);
                                tmpCanvas.drawText(Integer.toString(faceId),
                                        faceRect.left + faceRect.width() * textIndentFactor,
                                        faceRect.bottom - faceRect.height() * textIndentFactor,
                                        paint);

                                Bitmap faceBitmap = Bitmap.createBitmap(imageBitmap, faceRect.left, faceRect.top, faceRect.width(), faceRect.height());
                                classifyEmotions(faceBitmap, faceId);
                                faceId++;
                            }

                            mImageView.setImageBitmap(tmpBitmap);
                            ClassificationExpandableListAdapter adapter = new ClassificationExpandableListAdapter(mClassificationResult);
                            mClassificationExpandableListView.setAdapter(adapter);

                            if (faces.size() == 1) {
                                mClassificationExpandableListView.expandGroup(0);
                            }
                        } else {
                            Toast.makeText(MainActivity.this, getString(R.string.faceless), Toast.LENGTH_LONG).show();
                        }

                        setCalculationStatusUI(false);
                    }
                })
                .addOnFailureListener(new OnFailureListener() {
                    @Override
                    public void onFailure(@NonNull Exception e) {
                        e.printStackTrace();
                        setCalculationStatusUI(false);
                    }
                });
    }

    private void classifyEmotions(Bitmap imageBitmap, int faceId) {
        Map<String, Float> result = mClassifier.classify(imageBitmap, true);
        LinkedHashMap<String, Float> sortedResult = (LinkedHashMap<String, Float>) SortingHelper.sortByValues(result);

        ArrayList<String> reversedKeys = new ArrayList<>(sortedResult.keySet());
        Collections.reverse(reversedKeys);

        ArrayList<Pair<String, String>> faceGroup = new ArrayList<>();
        for (String key : reversedKeys) {
            String percentage = String.format("%.1f%%", sortedResult.get(key) * 100);
            faceGroup.add(new Pair<>(key, percentage));
        }

        String groupName = getString(R.string.face) + " " + faceId;
        mClassificationResult.put(groupName, faceGroup);
    }

    private Rect getInnerRect(Rect rect, int areaWidth, int areaHeight) {
        Rect innerRect = new Rect(rect);

        if (innerRect.top < 0) {
            innerRect.top = 0;
        }
        if (innerRect.left < 0) {
            innerRect.left = 0;
        }
        if (rect.bottom > areaHeight) {
            innerRect.bottom = areaHeight;
        }
        if (rect.right > areaWidth) {
            innerRect.right = areaWidth;
        }

        return innerRect;
    }

    private File createImageFile() throws IOException {
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "ER_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(imageFileName, ".jpg", storageDir);
        return image;
    }

    private void setCalculationStatusUI(boolean isCalculationRunning) {
        if (isCalculationRunning) {
            mClassificationProgressBar.setVisibility(ProgressBar.VISIBLE);
            mTakePhotoButton.setEnabled(false);
            mPickImageButton.setEnabled(false);
            mLiveEmotionButton.setEnabled(false);
        } else {
            mClassificationProgressBar.setVisibility(ProgressBar.INVISIBLE);
            mTakePhotoButton.setEnabled(true);
            mPickImageButton.setEnabled(true);
            mLiveEmotionButton.setEnabled(true);
        }
    }

    private void startLiveEmotionDetection() {
        Intent intent = new Intent(this, LiveEmotionActivity.class);
        startActivity(intent);
    }
}
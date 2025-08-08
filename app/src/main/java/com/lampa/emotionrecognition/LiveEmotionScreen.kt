// app/src/main/java/com/lampa/emotionrecognition/LiveEmotionScreen.kt
package com.lampa.emotionrecognition

import android.Manifest
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.Rect
import android.util.Log
import android.util.Size
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.lampa.emotionrecognition.classifiers.TFLiteImageClassifier
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import androidx.core.graphics.createBitmap

@Composable
fun LiveEmotionScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    // State for camera permission
    var hasCameraPermission by remember { mutableStateOf(false) }

    // State for UI
    val emotionText = remember { mutableStateOf("Point camera at a face") }
    var lensFacing by remember { mutableIntStateOf(CameraSelector.LENS_FACING_FRONT) }

    // Permission launcher
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
        onResult = { granted ->
            hasCameraPermission = granted
        }
    )

    // Request permission on launch
    LaunchedEffect(Unit) {
        permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    // Main UI layout
    Box(modifier = Modifier.fillMaxSize()) {
        if (hasCameraPermission) {
            CameraView(
                context = context,
                lifecycleOwner = lifecycleOwner,
                lensFacing = lensFacing,
                onEmotionDetected = { newEmotionText ->
                    emotionText.value = newEmotionText
                }
            )
        } else {
            Text("Camera permission is required.", modifier = Modifier.align(Alignment.Center))
        }

        // Bottom control panel
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Text(
                text = emotionText.value,
                color = Color.White,
                fontSize = 18.sp,
                textAlign = TextAlign.Center,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 16.dp)
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceAround
            ) {
                Button(onClick = onBack) {
                    Text("Back")
                }
                Button(onClick = {
                    lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
                        CameraSelector.LENS_FACING_BACK
                    } else {
                        CameraSelector.LENS_FACING_FRONT
                    }
                }) {
                    Text("Switch Camera")
                }
            }
        }
    }
}

@Composable
fun CameraView(
    context: Context,
    lifecycleOwner: LifecycleOwner,
    lensFacing: Int,
    onEmotionDetected: (String) -> Unit
) {
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    var cameraProvider: ProcessCameraProvider? by remember { mutableStateOf(null) }
    val previewView = remember { PreviewView(context) }

    // Classifier and Face Detector setup
    val emotionClassifier = remember {
        TFLiteImageClassifier(
            context.assets,
            "simple_classifier.tflite",
            context.resources.getStringArray(R.array.emotions)
        )
    }
    val faceDetector = remember {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .build()
        FaceDetection.getClient(options)
    }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    // Bind camera use cases in a LaunchedEffect
    LaunchedEffect(lensFacing, cameraProvider) {
        if (cameraProvider != null) {
            try {
                cameraProvider!!.unbindAll()
                val preview = Preview.Builder().build().also {
                    it.surfaceProvider = previewView.surfaceProvider
                }

                val imageAnalyzer = ImageAnalysis.Builder()
                    .setTargetResolution(Size(480, 640))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        it.setAnalyzer(cameraExecutor, EmotionAnalyzer(faceDetector, emotionClassifier, onEmotionDetected))
                    }

                val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
                cameraProvider!!.bindToLifecycle(lifecycleOwner, cameraSelector, preview, imageAnalyzer)
            } catch (e: Exception) {
                Log.e("CameraView", "Use case binding failed", e)
            }
        }
    }

    // Cleanup resources
    DisposableEffect(Unit) {
        onDispose {
            cameraExecutor.shutdown()
            faceDetector.close()
            emotionClassifier.close()
        }
    }

    // The actual camera preview
    AndroidView({ previewView }, modifier = Modifier.fillMaxSize()) {
        // This block is called when the view is created
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
        }, ContextCompat.getMainExecutor(context))
    }
}

private class EmotionAnalyzer(
    private val faceDetector: FaceDetector,
    private val emotionClassifier: TFLiteImageClassifier,
    private val onEmotionDetected: (String) -> Unit
) : ImageAnalysis.Analyzer {

    private var isProcessing = false

    @androidx.annotation.OptIn(androidx.camera.core.ExperimentalGetImage::class)
    override fun analyze(imageProxy: ImageProxy) {
        if (isProcessing) {
            imageProxy.close()
            return
        }
        isProcessing = true

        val mediaImage = imageProxy.image
        if (mediaImage != null) {
            val inputImage = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)

            faceDetector.process(inputImage)
                .addOnSuccessListener { faces ->
                    if (faces.isEmpty()) {
                        onEmotionDetected("No face detected")
                    } else {
                        val emotionText = StringBuilder()
                        faces.forEachIndexed { index, face ->
                            val faceBitmap = extractFaceBitmap(inputImage, face.boundingBox)
                            if (faceBitmap != null) {
                                val emotions = emotionClassifier.classify(faceBitmap, true)
                                val dominantEmotion = getDominantEmotion(emotions)
                                val confidence = emotions.getOrDefault(dominantEmotion, 0f) * 100
                                emotionText.append("Face ${index + 1}: $dominantEmotion (${"%.1f".format(confidence)}%)\n")
                            }
                        }
                        onEmotionDetected(emotionText.toString().trim())
                    }
                }
                .addOnFailureListener { e ->
                    Log.e("EmotionAnalyzer", "Face detection failed", e)
                    onEmotionDetected("Error detecting face")
                }
                .addOnCompleteListener {
                    isProcessing = false
                    imageProxy.close()
                }
        }
    }
}

// Helper functions (can be moved to a separate file)
@androidx.annotation.OptIn(androidx.camera.core.ExperimentalGetImage::class)
fun extractFaceBitmap(inputImage: InputImage, boundingBox: Rect): Bitmap? {
    val mediaImage = inputImage.mediaImage ?: return null
    val rotationDegrees = inputImage.rotationDegrees

    val imageBytes = imageToByteBuffer(mediaImage)
    val originalBitmap = yuvToRgb(imageBytes, mediaImage.width, mediaImage.height)

    val rotatedBitmap = if (rotationDegrees != 0) {
        val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        Bitmap.createBitmap(originalBitmap, 0, 0, originalBitmap.width, originalBitmap.height, matrix, true)
    } else {
        originalBitmap
    }

    val left = boundingBox.left.coerceAtLeast(0)
    val top = boundingBox.top.coerceAtLeast(0)
    val width = boundingBox.width().let { if (left + it > rotatedBitmap.width) rotatedBitmap.width - left else it }
    val height = boundingBox.height().let { if (top + it > rotatedBitmap.height) rotatedBitmap.height - top else it }

    return if (width > 0 && height > 0) {
        Bitmap.createBitmap(rotatedBitmap, left, top, width, height)
    } else {
        null
    }
}

fun getDominantEmotion(emotions: Map<String, Float>): String {
    return emotions.maxByOrNull { it.value }?.key ?: "Unknown"
}

private fun yuvToRgb(byteBuffer: ByteBuffer, width: Int, height: Int): Bitmap {
    byteBuffer.rewind()
    val data = ByteArray(byteBuffer.remaining())
    byteBuffer.get(data)
    val pixels = IntArray(width * height)
    val y = data
    for (i in pixels.indices) {
        val grey = y[i].toInt() and 0xFF
        pixels[i] = -0x1000000 or (grey shl 16) or (grey shl 8) or grey
    }
    val bitmap = createBitmap(width, height)
    bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
    return bitmap
}

private fun imageToByteBuffer(image: android.media.Image): ByteBuffer {
    val yBuffer = image.planes[0].buffer
    yBuffer.rewind()
    return yBuffer
}
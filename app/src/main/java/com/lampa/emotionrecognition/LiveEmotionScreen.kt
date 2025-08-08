package com.lampa.emotionrecognition

import android.Manifest
import android.content.Context
import android.graphics.Paint
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
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size as ComposeSize
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.lampa.emotionrecognition.classifiers.TFLiteImageClassifier
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

// Data class to hold all the information needed for drawing
data class FaceResult(
    val boundingBox: Rect,
    val emotion: String,
)

@Composable
fun LiveEmotionScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    var hasCameraPermission by remember { mutableStateOf(false) }

    // State for the results from the analyzer
    var faceResults by remember { mutableStateOf<List<FaceResult>>(emptyList()) }
    var analysisImageSize by remember { mutableStateOf(Size(0, 0)) }

    // State for the camera lens
    var lensFacing by remember { mutableStateOf(CameraSelector.LENS_FACING_FRONT) }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
        onResult = { granted -> hasCameraPermission = granted }
    )

    LaunchedEffect(Unit) {
        permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    Box(modifier = Modifier.fillMaxSize()) {
        if (hasCameraPermission) {
            CameraView(
                context = context,
                lensFacing = lensFacing,
                onResults = { results, imageSize ->
                    faceResults = results
                    analysisImageSize = imageSize
                }
            )
            FaceOverlay(
                faces = faceResults,
                sourceSize = analysisImageSize,
                lensFacing = lensFacing
            )
        } else {
            Text("Camera permission is required.", modifier = Modifier.align(Alignment.Center))
        }

        // Bottom control panel
        Row(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .padding(16.dp),
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
                faceResults = emptyList()
            }) {
                Text("Switch Camera")
            }
        }
    }
}

@Composable
fun CameraView(
    context: Context,
    lensFacing: Int,
    onResults: (List<FaceResult>, Size) -> Unit
) {
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    val previewView = remember { PreviewView(context) }

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

    LaunchedEffect(lensFacing) {
        val cameraProvider = cameraProviderFuture.get()
        bindCameraUseCases(
            cameraProvider,
            lifecycleOwner,
            previewView,
            lensFacing,
            cameraExecutor,
            faceDetector,
            emotionClassifier,
            onResults
        )
    }

    DisposableEffect(Unit) {
        onDispose {
            cameraExecutor.shutdown()
            faceDetector.close()
            emotionClassifier.close()
        }
    }

    AndroidView({ previewView }, modifier = Modifier.fillMaxSize())
}

private fun bindCameraUseCases(
    cameraProvider: ProcessCameraProvider,
    lifecycleOwner: LifecycleOwner,
    previewView: PreviewView,
    lensFacing: Int,
    cameraExecutor: java.util.concurrent.Executor,
    faceDetector: FaceDetector,
    emotionClassifier: TFLiteImageClassifier,
    onResults: (List<FaceResult>, Size) -> Unit
) {
    try {
        cameraProvider.unbindAll()
        val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

        // Ensure the PreviewView is attached to the window
        val preview = Preview.Builder()
            .setTargetResolution(Size(640, 480)) // Use a standard resolution
            .build()
            .also { it.setSurfaceProvider(previewView.surfaceProvider) }

        val imageAnalyzer = ImageAnalysis.Builder()
            .setTargetResolution(Size(640, 480)) // Match the preview resolution
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also {
                it.setAnalyzer(cameraExecutor, EmotionAnalyzer(faceDetector, emotionClassifier, onResults))
            }

        cameraProvider.bindToLifecycle(lifecycleOwner, cameraSelector, preview, imageAnalyzer)
    } catch (e: Exception) {
        Log.e("CameraView", "Use case binding failed", e)
    }
}

private class EmotionAnalyzer(
    private val faceDetector: FaceDetector,
    private val emotionClassifier: TFLiteImageClassifier,
    private val onResults: (List<FaceResult>, Size) -> Unit
) : ImageAnalysis.Analyzer {
    private var isProcessing = false

    @androidx.annotation.OptIn(androidx.camera.core.ExperimentalGetImage::class)
    override fun analyze(imageProxy: ImageProxy) {
        if (isProcessing) {
            imageProxy.close(); return
        }
        isProcessing = true

        val mediaImage = imageProxy.image
        // IMPORTANT: The analysis image is in landscape orientation, so width > height.
        val analysisSize = Size(imageProxy.width, imageProxy.height)

        if (mediaImage != null) {
            val inputImage = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)
            faceDetector.process(inputImage)
                .addOnSuccessListener { faces ->
                    val results = faces.mapNotNull { face ->
                        extractFaceBitmap(inputImage, face.boundingBox)?.let { faceBitmap ->
                            val emotions = emotionClassifier.classify(faceBitmap, true)
                            val dominantEmotion = getDominantEmotion(emotions)
                            val confidence = emotions.getOrDefault(dominantEmotion, 0f) * 100
                            FaceResult(
                                boundingBox = face.boundingBox,
                                emotion = "$dominantEmotion (${"%.1f".format(confidence)}%)"
                            )
                        }
                    }
                    onResults(results, analysisSize)
                }
                .addOnFailureListener { e -> Log.e("EmotionAnalyzer", "Face detection failed", e) }
                .addOnCompleteListener {
                    isProcessing = false
                    imageProxy.close()
                }
        } else {
            imageProxy.close() // Close if mediaImage is null
        }
    }
}

@Composable
fun FaceOverlay(
    faces: List<FaceResult>,
    sourceSize: Size,
    lensFacing: Int
) {
    val textPaint = remember {
        Paint().apply {
            color = android.graphics.Color.WHITE
            textSize = 42f
            textAlign = Paint.Align.CENTER
        }
    }

    Canvas(modifier = Modifier.fillMaxSize()) {
        // Guard against division by zero
        if (sourceSize.width == 0 || sourceSize.height == 0) return@Canvas

        // This is the size of the Composable Canvas. It's in portrait.
        val canvasWidth = size.width
        val canvasHeight = size.height

        // The camera analysis provides a landscape image (width > height).
        // We need to map coordinates from this landscape source to the portrait canvas.

        // How much we need to scale the X and Y dimensions from the source to the canvas.
        val widthScaleFactor = canvasWidth / sourceSize.height.toFloat()
        val heightScaleFactor = canvasHeight / sourceSize.width.toFloat()

        for (face in faces) {
            val box = face.boundingBox

            // Transform the coordinates from the landscape source to the portrait canvas.
            val left = box.top * widthScaleFactor
            val top = box.left * heightScaleFactor
            val right = box.bottom * widthScaleFactor
            val bottom = box.right * heightScaleFactor

            // For the front camera, the image is mirrored. We need to flip the X-coordinates.
            val mirroredLeft = if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
                canvasWidth - right
            } else {
                left
            }
            val mirroredRight = if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
                canvasWidth - left
            } else {
                right
            }

            // Draw the bounding box. Use min/max to handle the mirrored case correctly.
            drawRect(
                color = Color.Green,
                topLeft = Offset(min(mirroredLeft, mirroredRight), top),
                size = ComposeSize(abs(mirroredRight - mirroredLeft), bottom - top),
                style = Stroke(width = 2.dp.toPx())
            )

            // Draw the emotion text above the box
            drawContext.canvas.nativeCanvas.drawText(
                face.emotion,
                (mirroredLeft + mirroredRight) / 2, // Center the text horizontally
                top - 15,                         // Position text slightly above the box
                textPaint
            )
        }
    }
}


// --- Helper Functions (No changes needed here from last time) ---
@androidx.annotation.OptIn(androidx.camera.core.ExperimentalGetImage::class)
fun extractFaceBitmap(inputImage: InputImage, boundingBox: Rect): android.graphics.Bitmap? {
    val mediaImage = inputImage.mediaImage ?: return null
    val rotationDegrees = inputImage.rotationDegrees

    // This conversion is a fallback. For better performance, a direct YUV_420_888 to ARGB
    // converter would be ideal, but this is more universally compatible.
    val yBuffer = mediaImage.planes[0].buffer.apply { rewind() }
    val uBuffer = mediaImage.planes[1].buffer.apply { rewind() }
    val vBuffer = mediaImage.planes[2].buffer.apply { rewind() }
    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()
    val nv21 = ByteArray(ySize + uSize + vSize)
    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)
    val yuvImage = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, mediaImage.width, mediaImage.height, null)
    val out = java.io.ByteArrayOutputStream()
    yuvImage.compressToJpeg(android.graphics.Rect(0, 0, mediaImage.width, mediaImage.height), 100, out)
    val imageBytes = out.toByteArray()
    val originalBitmap = android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

    val rotatedBitmap = if (rotationDegrees != 0) {
        val matrix = android.graphics.Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        android.graphics.Bitmap.createBitmap(originalBitmap, 0, 0, originalBitmap.width, originalBitmap.height, matrix, true)
    } else {
        originalBitmap
    }

    val left = max(0, boundingBox.left)
    val top = max(0, boundingBox.top)
    val width = min(rotatedBitmap.width - left, boundingBox.width())
    val height = min(rotatedBitmap.height - top, boundingBox.height())

    return if (width > 0 && height > 0) {
        android.graphics.Bitmap.createBitmap(rotatedBitmap, left, top, width, height)
    } else {
        null
    }
}

fun getDominantEmotion(emotions: Map<String, Float>): String {
    return emotions.maxByOrNull { it.value }?.key ?: "Unknown"
}
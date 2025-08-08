package com.lampa.emotionrecognition

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color as GraphicsColor
import android.graphics.Paint
import android.graphics.Rect
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size as ComposeSize
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.LifecycleOwner
import androidx.navigation.NavController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.lampa.emotionrecognition.classifiers.TFLiteImageClassifier
import com.lampa.emotionrecognition.ui.theme.EmotionRecognitionTheme
import com.lampa.emotionrecognition.utils.ImageUtils
import com.lampa.emotionrecognition.utils.SortingHelper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.Executors
import kotlin.coroutines.suspendCoroutine
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

class MainActivity : ComponentActivity() {

    private lateinit var classifier: TFLiteImageClassifier

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        classifier = TFLiteImageClassifier(
            assets,
            "simple_classifier.tflite",
            resources.getStringArray(R.array.emotions)
        )

        setContent {
            EmotionRecognitionTheme {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    // Set up the NavController and NavHost
                    val navController = rememberNavController()
                    NavHost(navController = navController, startDestination = "main") {
                        composable("main") {
                            MainScreen(navController = navController, classifier = classifier)
                        }
                        composable("live_emotion") {
                            LiveEmotionScreen(navController = navController)
                        }
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::classifier.isInitialized) {
            classifier.close()
        }
    }
}

// --- Screen 1: Main Screen for Image Analysis ---
@Composable
fun MainScreen(navController: NavController, classifier: TFLiteImageClassifier) {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()

    var imageUri by remember { mutableStateOf<Uri?>(null) }
    var bitmap by remember { mutableStateOf<Bitmap?>(null) }
    var classificationResult by remember { mutableStateOf<Map<String, List<Pair<String, String>>>>(emptyMap()) }
    var isLoading by remember { mutableStateOf(false) }
    var tempPhotoUri by remember { mutableStateOf<Uri?>(null) }

    val galleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent(),
        onResult = { uri -> if (uri != null) imageUri = uri }
    )

    val cameraLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicture(),
        onResult = { success -> if (success) imageUri = tempPhotoUri }
    )

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
        onResult = { isGranted ->
            if (!isGranted) Toast.makeText(context, "Permission Denied", Toast.LENGTH_SHORT).show()
        }
    )

    LaunchedEffect(imageUri) {
        imageUri?.let { uri ->
            isLoading = true
            classificationResult = emptyMap()
            coroutineScope.launch {
                val processedBitmap = processImage(context, uri)
                bitmap = processedBitmap
                processedBitmap?.let { bmp ->
                    val results = detectAndClassify(bmp, classifier)
                    bitmap = results.first
                    classificationResult = results.second
                }
                isLoading = false
            }
        }
    }

    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Box(
            modifier = Modifier.fillMaxWidth().height(240.dp).background(MaterialTheme.colorScheme.surfaceVariant),
            contentAlignment = Alignment.Center
        ) {
            bitmap?.let { Image(bitmap = it.asImageBitmap(), contentDescription = "Selected", contentScale = ContentScale.Fit) }
                ?: Text("Select an image or take a photo")
            if (isLoading) CircularProgressIndicator()
        }
        Spacer(modifier = Modifier.height(16.dp))
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { galleryLauncher.launch("image/*") }, modifier = Modifier.weight(1f)) {
                Text("Pick Image")
            }
            Button(onClick = {
                if (hasCameraPermission(context)) {
                    val uri = createImageUri(context)
                    tempPhotoUri = uri
                    cameraLauncher.launch(uri)
                } else {
                    permissionLauncher.launch(Manifest.permission.CAMERA)
                }
            }, modifier = Modifier.weight(1f)) {
                Text("Take Photo")
            }
        }
        Button(
            onClick = { navController.navigate("live_emotion") }, // Navigate to the live screen
            modifier = Modifier.fillMaxWidth().padding(top = 8.dp)
        ) {
            Text("Live Emotion Detection")
        }
        Divider(modifier = Modifier.padding(vertical = 16.dp))
        ClassificationResultList(results = classificationResult)
    }
}

// --- Screen 2: Live Emotion Detection Screen ---
@Composable
fun LiveEmotionScreen(navController: NavController) {
    val context = LocalContext.current
    var hasCameraPermission by remember { mutableStateOf(false) }

    var faceResults by remember { mutableStateOf<List<FaceResult>>(emptyList()) }
    var analysisImageSize by remember { mutableStateOf(Size(0, 0)) }
    var lensFacing by remember { mutableStateOf(CameraSelector.LENS_FACING_FRONT) }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
        onResult = { granted -> hasCameraPermission = granted }
    )

    LaunchedEffect(Unit) {
        if (!hasCameraPermission(context)) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        } else {
            hasCameraPermission = true
        }
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

        Row(
            modifier = Modifier.align(Alignment.BottomCenter).fillMaxWidth().padding(16.dp),
            horizontalArrangement = Arrangement.SpaceAround
        ) {
            Button(onClick = { navController.popBackStack() }) { // Navigate back
                Text("Back")
            }
            Button(onClick = {
                lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT) CameraSelector.LENS_FACING_BACK else CameraSelector.LENS_FACING_FRONT
                faceResults = emptyList()
            }) {
                Text("Switch Camera")
            }
        }
    }
}


// --- All other composables and helper functions remain below ---

data class FaceResult(val boundingBox: Rect, val emotion: String)

@Composable
fun ClassificationResultList(results: Map<String, List<Pair<String, String>>>) {
    if (results.isEmpty()) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text("Results will be shown here")
        }
    } else {
        LazyColumn(modifier = Modifier.fillMaxSize()) {
            items(results.entries.toList()) { (faceTitle, emotions) ->
                Column {
                    Text(
                        text = faceTitle, fontSize = 18.sp, fontWeight = FontWeight.Bold,
                        modifier = Modifier.fillMaxWidth().background(MaterialTheme.colorScheme.primaryContainer).padding(8.dp)
                    )
                    emotions.forEach { (emotion, probability) ->
                        Row(
                            modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(text = emotion, fontSize = 16.sp)
                            Text(text = probability, fontSize = 16.sp)
                        }
                    }
                }
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
        FaceDetection.getClient(FaceDetectorOptions.Builder().setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST).build())
    }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    LaunchedEffect(lensFacing) {
        val cameraProvider = cameraProviderFuture.get()
        bindCameraUseCases(cameraProvider, lifecycleOwner, previewView, lensFacing, cameraExecutor, faceDetector, emotionClassifier, onResults)
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
        val preview = Preview.Builder().setTargetResolution(Size(640, 480)).build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
        val imageAnalyzer = ImageAnalysis.Builder().setTargetResolution(Size(640, 480)).setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()
            .also { it.setAnalyzer(cameraExecutor, EmotionAnalyzer(faceDetector, emotionClassifier, onResults)) }
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
        val analysisSize = Size(imageProxy.width, imageProxy.height)
        if (mediaImage != null) {
            val inputImage = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)
            faceDetector.process(inputImage)
                .addOnSuccessListener { faces ->
                    val results = faces.mapNotNull { face ->
                        val scaledBox = scaleBoundingBox(face.boundingBox, 1.4f)
                        extractFaceBitmap(inputImage, scaledBox)?.let { faceBitmap ->
                            val emotions = emotionClassifier.classify(faceBitmap, true)
                            val dominantEmotion = getDominantEmotion(emotions)
                            val confidence = emotions.getOrDefault(dominantEmotion, 0f) * 100
                            FaceResult(scaledBox, "$dominantEmotion (${"%.1f".format(confidence)}%)")
                        }
                    }
                    onResults(results, analysisSize)
                }
                .addOnFailureListener { e -> Log.e("EmotionAnalyzer", "Face detection failed", e) }
                .addOnCompleteListener { isProcessing = false; imageProxy.close() }
        } else {
            imageProxy.close()
        }
    }
}

@Composable
fun FaceOverlay(faces: List<FaceResult>, sourceSize: Size, lensFacing: Int) {
    val textPaint = remember { Paint().apply { color = GraphicsColor.WHITE; textSize = 42f; textAlign = Paint.Align.CENTER } }
    Canvas(modifier = Modifier.fillMaxSize()) {
        if (sourceSize.width == 0 || sourceSize.height == 0) return@Canvas
        val canvasWidth = size.width
        val canvasHeight = size.height
        val widthScaleFactor = canvasWidth / sourceSize.height.toFloat()
        val heightScaleFactor = canvasHeight / sourceSize.width.toFloat()
        for (face in faces) {
            val box = face.boundingBox
            val left = box.top * widthScaleFactor
            val top = box.left * heightScaleFactor
            val right = box.bottom * widthScaleFactor
            val bottom = box.right * heightScaleFactor
            val mirroredLeft = if (lensFacing == CameraSelector.LENS_FACING_FRONT) canvasWidth - right else left
            val mirroredRight = if (lensFacing == CameraSelector.LENS_FACING_FRONT) canvasWidth - left else right
            drawRect(Color.Green, Offset(min(mirroredLeft, mirroredRight), top), ComposeSize(abs(mirroredRight - mirroredLeft), bottom - top), style = Stroke(width = 2.dp.toPx()))
            drawContext.canvas.nativeCanvas.drawText(face.emotion, (mirroredLeft + mirroredRight) / 2, top - 15, textPaint)
        }
    }
}

// All other helper functions (processImage, detectAndClassify, hasCameraPermission, etc.)
// ...
private suspend fun processImage(context: Context, uri: Uri): Bitmap? {
    return withContext(Dispatchers.IO) {
        try {
            val originalBitmap = MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
            val rotatedBitmap = ImageUtils.rotateToNormalOrientation(context.contentResolver, originalBitmap, uri)
            val scaleFactor = 480f / max(rotatedBitmap.width, rotatedBitmap.height)
            Bitmap.createScaledBitmap(rotatedBitmap, (rotatedBitmap.width * scaleFactor).toInt(), (rotatedBitmap.height * scaleFactor).toInt(), true)
        } catch (e: Exception) {
            e.printStackTrace(); null
        }
    }
}

private suspend fun detectAndClassify(bitmap: Bitmap, classifier: TFLiteImageClassifier): Pair<Bitmap, Map<String, List<Pair<String, String>>>> {
    return withContext(Dispatchers.Default) {
        val faceDetector = FaceDetection.getClient(FaceDetectorOptions.Builder().setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE).build())
        val inputImage = InputImage.fromBitmap(bitmap, 0)
        val finalResults = mutableMapOf<String, List<Pair<String, String>>>()
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val paint = Paint().apply { color = GraphicsColor.GREEN; strokeWidth = 2f; style = Paint.Style.STROKE; textSize = 48f }
        val faces = suspendCoroutine<List<Face>> { cont -> faceDetector.process(inputImage).addOnSuccessListener { cont.resumeWith(Result.success(it)) }.addOnFailureListener { cont.resumeWith(Result.failure(it)) } }
        if (faces.isNotEmpty()) {
            faces.forEachIndexed { index, face ->
                canvas.drawRect(face.boundingBox, paint)
                val faceBitmap = Bitmap.createBitmap(bitmap, face.boundingBox.left.coerceAtLeast(0), face.boundingBox.top.coerceAtLeast(0), face.boundingBox.width().coerceAtMost(bitmap.width - face.boundingBox.left.coerceAtLeast(0)), face.boundingBox.height().coerceAtMost(bitmap.height - face.boundingBox.top.coerceAtLeast(0)))
                val emotions = classifier.classify(faceBitmap, true)
                val sortedEmotions = (SortingHelper.sortByValues(emotions) as LinkedHashMap<String, Float>).entries.reversed().map { it.key to String.format("%.1f%%", it.value * 100) }
                finalResults["Face ${index + 1}"] = sortedEmotions
            }
        }
        faceDetector.close()
        Pair(mutableBitmap, finalResults)
    }
}

private fun hasCameraPermission(context: Context): Boolean = ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
private fun createImageUri(context: Context): Uri {
    val imageFile = File.createTempFile("JPEG_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())}_", ".jpg", context.getExternalFilesDir(Environment.DIRECTORY_PICTURES))
    return FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", imageFile)
}
private fun scaleBoundingBox(rect: Rect, scale: Float): Rect {
    val newWidth = rect.width() * scale
    val newHeight = rect.height() * scale
    return Rect((rect.centerX() - newWidth / 2).toInt(), (rect.centerY() - newHeight / 2).toInt(), (rect.centerX() + newWidth / 2).toInt(), (rect.centerY() + newHeight / 2).toInt())
}
@androidx.annotation.OptIn(androidx.camera.core.ExperimentalGetImage::class)
fun extractFaceBitmap(inputImage: InputImage, boundingBox: Rect): android.graphics.Bitmap? {
    val mediaImage = inputImage.mediaImage ?: return null
    val rotationDegrees = inputImage.rotationDegrees
    val yBuffer = mediaImage.planes[0].buffer.apply { rewind() }
    val uBuffer = mediaImage.planes[1].buffer.apply { rewind() }
    val vBuffer = mediaImage.planes[2].buffer.apply { rewind() }
    val ySize = yBuffer.remaining(); val uSize = uBuffer.remaining(); val vSize = vBuffer.remaining()
    val nv21 = ByteArray(ySize + uSize + vSize)
    yBuffer.get(nv21, 0, ySize); vBuffer.get(nv21, ySize, vSize); uBuffer.get(nv21, ySize + vSize, uSize)
    val yuvImage = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, mediaImage.width, mediaImage.height, null)
    val out = java.io.ByteArrayOutputStream()
    yuvImage.compressToJpeg(android.graphics.Rect(0, 0, mediaImage.width, mediaImage.height), 100, out)
    val imageBytes = out.toByteArray()
    val originalBitmap = android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    val rotatedBitmap = if (rotationDegrees != 0) {
        val matrix = android.graphics.Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        android.graphics.Bitmap.createBitmap(originalBitmap, 0, 0, originalBitmap.width, originalBitmap.height, matrix, true)
    } else originalBitmap
    val left = max(0, boundingBox.left); val top = max(0, boundingBox.top)
    val width = min(rotatedBitmap.width - left, boundingBox.width()); val height = min(rotatedBitmap.height - top, boundingBox.height())
    return if (width > 0 && height > 0) android.graphics.Bitmap.createBitmap(rotatedBitmap, left, top, width, height) else null
}
fun getDominantEmotion(emotions: Map<String, Float>): String = emotions.maxByOrNull { it.value }?.key ?: "Unknown"
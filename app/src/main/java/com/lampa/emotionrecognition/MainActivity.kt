package com.lampa.emotionrecognition

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
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
import kotlin.coroutines.suspendCoroutine
import androidx.core.graphics.scale

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
                    MainScreen(classifier)
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

@Composable
fun MainScreen(classifier: TFLiteImageClassifier) {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()

    var imageUri by remember { mutableStateOf<Uri?>(null) }
    var bitmap by remember { mutableStateOf<Bitmap?>(null) }
    var classificationResult by remember { mutableStateOf<Map<String, List<Pair<String, String>>>>(emptyMap()) }
    var isLoading by remember { mutableStateOf(false) }

    // This state is only needed to receive the result from the camera
    var tempPhotoUri by remember { mutableStateOf<Uri?>(null) }

    val galleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent(),
        onResult = { uri ->
            if (uri != null) {
                imageUri = uri
            }
        }
    )

    val cameraLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicture(),
        onResult = { success ->
            if (success) {
                // Use the URI that was saved before launching the camera
                imageUri = tempPhotoUri
            }
        }
    )

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
        onResult = { isGranted ->
            if (isGranted) {
                Toast.makeText(context, "Permission Granted", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(context, "Permission Denied", Toast.LENGTH_SHORT).show()
            }
        }
    )

    // Effect to process the image whenever the URI changes
    LaunchedEffect(imageUri) {
        imageUri?.let { uri ->
            isLoading = true
            classificationResult = emptyMap()
            coroutineScope.launch {
                val processedBitmap = processImage(context, uri)
                bitmap = processedBitmap
                processedBitmap?.let { bmp ->
                    val results = detectAndClassify(bmp, classifier)
                    bitmap = results.first // The bitmap might have face boxes drawn on it
                    classificationResult = results.second
                }
                isLoading = false
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Image Display Area
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(240.dp)
                .background(MaterialTheme.colorScheme.surfaceVariant),
            contentAlignment = Alignment.Center
        ) {
            bitmap?.let {
                Image(
                    bitmap = it.asImageBitmap(),
                    contentDescription = "Selected Image",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Fit
                )
            } ?: Text("Select an image or take a photo")

            if (isLoading) {
                CircularProgressIndicator()
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Action Buttons
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(onClick = { galleryLauncher.launch("image/*") }, modifier = Modifier.weight(1f)) {
                Text("Pick Image")
            }
            Button(
                onClick = {
                    if (hasCameraPermission(context)) {
                        // --- THE FIX IS HERE ---
                        // 1. Create a non-nullable local variable for the URI
                        val uri = createImageUri(context)
                        // 2. Save it to the state variable for the callback
                        tempPhotoUri = uri
                        // 3. Launch the camera with the guaranteed non-null local variable
                        cameraLauncher.launch(uri)
                    } else {
                        permissionLauncher.launch(Manifest.permission.CAMERA)
                    }
                },
                modifier = Modifier.weight(1f)
            ) {
                Text("Take Photo")
            }
        }

        Button(
            onClick = {
                context.startActivity(Intent(context, LiveEmotionComposeActivity::class.java))
            },
            modifier = Modifier.fillMaxWidth().padding(top = 8.dp)
        ) {
            Text("Live Emotion Detection")
        }


        Divider(modifier = Modifier.padding(vertical = 16.dp))

        // Results Display
        ClassificationResultList(results = classificationResult)
    }
}

// The rest of the file (ClassificationResultList and helper functions) remains the same.
// ... (omitted for brevity, no changes needed below this line) ...

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
                        text = faceTitle,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(MaterialTheme.colorScheme.primaryContainer)
                            .padding(8.dp)
                    )
                    emotions.forEach { (emotion, probability) ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 16.dp, vertical = 4.dp),
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


// --- Helper Functions ---

private suspend fun processImage(context: Context, uri: Uri): Bitmap? {
    return withContext(Dispatchers.IO) {
        try {
            val originalBitmap = MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
            val rotatedBitmap = ImageUtils.rotateToNormalOrientation(context.contentResolver, originalBitmap, uri)

            // Scale the bitmap
            val scaleFactor = 480f / maxOf(rotatedBitmap.width, rotatedBitmap.height)
            rotatedBitmap.scale(
                (rotatedBitmap.width * scaleFactor).toInt(),
                (rotatedBitmap.height * scaleFactor).toInt()
            )
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }
}

private suspend fun detectAndClassify(
    bitmap: Bitmap,
    classifier: TFLiteImageClassifier
): Pair<Bitmap, Map<String, List<Pair<String, String>>>> {
    return withContext(Dispatchers.Default) {
        val faceDetector = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .build().let { FaceDetection.getClient(it) }

        val inputImage = InputImage.fromBitmap(bitmap, 0)
        val finalResults = mutableMapOf<String, List<Pair<String, String>>>()

        // Create a mutable bitmap to draw on
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val paint = Paint().apply {
            color = Color.GREEN
            strokeWidth = 2f
            style = Paint.Style.STROKE
            textSize = 48f
        }

        val faces = suspendCoroutine<List<Face>> { continuation ->
            faceDetector.process(inputImage)
                .addOnSuccessListener { continuation.resumeWith(Result.success(it)) }
                .addOnFailureListener { continuation.resumeWith(Result.failure(it)) }
        }

        if (faces.isNotEmpty()) {
            faces.forEachIndexed { index, face ->
                // Draw rectangle on the mutable bitmap
                canvas.drawRect(face.boundingBox, paint)

                // Crop face and classify
                val faceBitmap = Bitmap.createBitmap(
                    bitmap,
                    face.boundingBox.left.coerceAtLeast(0),
                    face.boundingBox.top.coerceAtLeast(0),
                    face.boundingBox.width().coerceAtMost(bitmap.width - face.boundingBox.left.coerceAtLeast(0)),
                    face.boundingBox.height().coerceAtMost(bitmap.height - face.boundingBox.top.coerceAtLeast(0))
                )

                val emotions = classifier.classify(faceBitmap, true)
                val sortedEmotions = (SortingHelper.sortByValues(emotions) as LinkedHashMap<String, Float>)
                    .entries
                    .reversed()
                    .map { entry -> entry.key to String.format("%.1f%%", entry.value * 100) }

                finalResults["Face ${index + 1}"] = sortedEmotions
            }
        }
        faceDetector.close()
        Pair(mutableBitmap, finalResults)
    }
}

private fun hasCameraPermission(context: Context): Boolean {
    return ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
}

private fun createImageUri(context: Context): Uri {
    val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
    val imageFile = File.createTempFile(
        "JPEG_${timeStamp}_",
        ".jpg",
        context.getExternalFilesDir(Environment.DIRECTORY_PICTURES)
    )
    return FileProvider.getUriForFile(
        context,
        "${context.packageName}.fileprovider",
        imageFile
    )
}
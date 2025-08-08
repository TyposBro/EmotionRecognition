package com.lampa.emotionrecognition

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent

class LiveEmotionComposeActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            // Your app's theme would wrap this
            LiveEmotionScreen(onBack = { finish() })
        }
    }
}
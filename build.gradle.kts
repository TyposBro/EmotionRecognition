// build.gradle.kts (Project level)
// This file no longer uses the 'libs' alias, which fixes the error.
plugins {
    id("com.android.application") version "8.12.0" apply false
    id("org.jetbrains.kotlin.android") version "1.9.10" apply false
}

// You can add a task to clean the build directory, which is good practice.
task("clean", type = Delete::class) {
    delete(rootProject.buildDir)
}
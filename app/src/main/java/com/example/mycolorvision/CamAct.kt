package com.example.mycolorvision

import android.app.Activity
import android.content.Intent
import android.graphics.*
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class CamAct : AppCompatActivity() {

    private lateinit var interpreter: Interpreter
    private val TAG = "CamAct"

    private var currentBitmap: Bitmap? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_cam)

        try {
            interpreter = Interpreter(loadModelFile())

            Log.d(TAG, "Número de entradas: ${interpreter.inputTensorCount}")
            Log.d(TAG, "Número de salidas: ${interpreter.outputTensorCount}")

            getTensorInfo()

        } catch (e: Exception) {
            Log.e(TAG, "Error cargando modelo: ${e.message}")
            Toast.makeText(this, "Error al cargar el modelo", Toast.LENGTH_LONG).show()
            e.printStackTrace()
        }

        val btnCamara = findViewById<Button>(R.id.btnCamara)
        val btnDeute = findViewById<Button>(R.id.botondeute)
        val btnMono = findViewById<Button>(R.id.botonmono)
        val imageView = findViewById<ImageView>(R.id.imageView)

        btnCamara.setOnClickListener {
            startForResult.launch(Intent(MediaStore.ACTION_IMAGE_CAPTURE))
        }

        btnDeute.setOnClickListener {
            val bmp = currentBitmap
            if (bmp == null) {
                Toast.makeText(this, "No hay imagen para procesar", Toast.LENGTH_SHORT).show()
            } else {
                try {
                    val transformed = applyDeuteranopia(bmp)
                    currentBitmap = transformed
                    imageView.setImageBitmap(transformed)
                    Toast.makeText(this, "Aplicado filtro: Deuteranopia", Toast.LENGTH_SHORT).show()
                } catch (ex: Exception) {
                    Log.e(TAG, "Error aplicando deuteranopia: ${ex.message}")
                    Toast.makeText(this, "Error aplicando filtro", Toast.LENGTH_SHORT).show()
                }
            }
        }

        btnMono.setOnClickListener {
            val bmp = currentBitmap
            if (bmp == null) {
                Toast.makeText(this, "No hay imagen para procesar", Toast.LENGTH_SHORT).show()
            } else {
                try {
                    val transformed = applyMonochrome(bmp, 128)
                    currentBitmap = transformed
                    imageView.setImageBitmap(transformed)
                    Toast.makeText(this, "Aplicado filtro: Blanco y Negro", Toast.LENGTH_SHORT).show()
                } catch (ex: Exception) {
                    Log.e(TAG, "Error aplicando blanco y negro: ${ex.message}")
                    Toast.makeText(this, "Error aplicando filtro", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun loadModelFile(): MappedByteBuffer {
        val assetFileDescriptor = assets.openFd("best_float32.tflite")
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun getTensorInfo() {
        try {
            Log.d(TAG, "Input count: ${interpreter.inputTensorCount}, Output count: ${interpreter.outputTensorCount}")
            for (i in 0 until interpreter.inputTensorCount) {
                val t = interpreter.getInputTensor(i)
                Log.d(TAG, "Input $i shape: ${t.shape().contentToString()}, dtype: ${t.dataType()}")
            }
            for (i in 0 until interpreter.outputTensorCount) {
                val t = interpreter.getOutputTensor(i)
                Log.d(TAG, "Output $i shape: ${t.shape().contentToString()}, dtype: ${t.dataType()}")
            }
        } catch (ex: Exception) {
            Log.e(TAG, "Error leyendo tensor info: ${ex.message}")
        }
    }

    // PREPROCESO

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val inputShape = interpreter.getInputTensor(0).shape()
        val isNCHW = (inputShape.size == 4 && inputShape[1] == 3)
        val height = if (isNCHW) inputShape[2] else inputShape[1]
        val width = if (isNCHW) inputShape[3] else inputShape[2]
        val channels = if (isNCHW) inputShape[1] else inputShape[3]

        val resized = Bitmap.createScaledBitmap(bitmap, width, height, true)

        val byteBuffer = ByteBuffer.allocateDirect(4 * width * height * channels)
        byteBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(width * height)
        resized.getPixels(intValues, 0, width, 0, 0, width, height)

        if (!isNCHW) {
            var p = 0
            for (y in 0 until height) {
                for (x in 0 until width) {
                    val v = intValues[p++]
                    byteBuffer.putFloat(((v shr 16) and 0xFF) / 255.0f)
                    byteBuffer.putFloat(((v shr 8) and 0xFF) / 255.0f)
                    byteBuffer.putFloat((v and 0xFF) / 255.0f)
                }
            }
        } else {
            for (c in 0 until channels) {
                for (y in 0 until height) {
                    for (x in 0 until width) {
                        val idx = y * width + x
                        val v = intValues[idx]
                        val value = when (c) {
                            0 -> ((v shr 16) and 0xFF) / 255.0f
                            1 -> ((v shr 8) and 0xFF) / 255.0f
                            else -> (v and 0xFF) / 255.0f
                        }
                        byteBuffer.putFloat(value)
                    }
                }
            }
        }

        byteBuffer.rewind() // IMPORTANTE luego me vuelve a fallar xD
        return byteBuffer
    }

    private fun runInference(bitmap: Bitmap): Bitmap {
        try {
            Log.d(TAG, "Iniciando inferencia...")
            val inputBuffer = preprocessImage(bitmap)

            val outputCount = interpreter.outputTensorCount

            if (outputCount == 1) {
                val outShape = interpreter.getOutputTensor(0).shape()
                val batch = outShape[0]
                val detCount = outShape[1]
                val detSize = outShape[2]

                val output = Array(batch) { Array(detCount) { FloatArray(detSize) } }
                interpreter.run(inputBuffer, output)
                Log.d(TAG, "Inferencia completada (1 output)")

                // Guardar imagen original
                val resultBitmap = drawYOLODetections(bitmap, output[0])
                currentBitmap = resultBitmap
                return resultBitmap

            } else {
                val outputs = mutableMapOf<Int, Any>()
                for (i in 0 until outputCount) {
                    val s = interpreter.getOutputTensor(i).shape()
                    val b = s[0]
                    val n = s[1]
                    val m = s[2]
                    val arr = Array(b) { Array(n) { FloatArray(m) } }
                    outputs[i] = arr
                }

                interpreter.runForMultipleInputsOutputs(arrayOf<Any>(inputBuffer), outputs)
                Log.d(TAG, "Inferencia completada (múltiples outputs)")

                // Concatenar lo del este batch 0
                val listDetections = mutableListOf<FloatArray>()
                for (i in 0 until outputCount) {
                    val arr = outputs[i] as Array<Array<FloatArray>>
                    for (row in arr[0]) listDetections.add(row)
                }

                if (listDetections.isEmpty()) {
                    Log.w(TAG, "No se obtuvieron detecciones")
                    currentBitmap = bitmap
                    return bitmap
                }

                val combined = Array(listDetections.size) { FloatArray(listDetections[0].size) }
                for (i in listDetections.indices) combined[i] = listDetections[i]

                val resultBitmap = drawYOLODetections(bitmap, combined)
                currentBitmap = resultBitmap
                return resultBitmap
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error en inferencia: ${e.message}")
            e.printStackTrace()
            Toast.makeText(this, "Error al procesar imagen: ${e.message}", Toast.LENGTH_LONG).show()
            currentBitmap = bitmap
            return bitmap
        }
    }

    // DIBUJADO
    private fun drawYOLODetections(bitmap: Bitmap, output: Array<FloatArray>): Bitmap {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)

        val paint = Paint().apply {
            color = Color.RED
            strokeWidth = 8f
            style = Paint.Style.STROKE
        }

        val textPaint = Paint().apply {
            color = Color.RED
            textSize = 50f
            style = Paint.Style.FILL
        }

        val bgPaint = Paint().apply {
            color = Color.WHITE
            style = Paint.Style.FILL
        }

        val threshold = 0.25f
        var detectionCount = 0

        for (detection in output) {
            if (detection.size < 6) continue

            val confidence = detection[4]

            if (confidence >= threshold) {
                var maxClassScore = 0f
                var maxClassIndex = 0

                for (i in 5 until detection.size) {
                    if (detection[i] > maxClassScore) {
                        maxClassScore = detection[i]
                        maxClassIndex = i - 5
                    }
                }

                val finalScore = confidence * maxClassScore

                if (finalScore >= threshold) {
                    detectionCount++

                    val centerX = detection[0] * bitmap.width
                    val centerY = detection[1] * bitmap.height
                    val width = detection[2] * bitmap.width
                    val height = detection[3] * bitmap.height

                    val left = centerX - width / 2
                    val top = centerY - height / 2
                    val right = centerX + width / 2
                    val bottom = centerY + height / 2

                    canvas.drawRect(left, top, right, bottom, paint)

                    val label = "Clase $maxClassIndex: ${String.format("%.2f", finalScore)}"

                    val textBounds = android.graphics.Rect()
                    textPaint.getTextBounds(label, 0, label.length, textBounds)
                    canvas.drawRect(
                        left,
                        top - textBounds.height() - 20,
                        left + textBounds.width() + 20,
                        top,
                        bgPaint
                    )

                    canvas.drawText(label, left + 10, top - 10, textPaint)
                }
            }
        }

        Log.d(TAG, "Detecciones encontradas: $detectionCount")
        Toast.makeText(this, "Detecciones: $detectionCount", Toast.LENGTH_SHORT).show()

        return mutableBitmap
    }

    private fun applyDeuteranopia(source: Bitmap): Bitmap {
        val m00 = 0.625f; val m01 = 0.375f; val m02 = 0f
        val m10 = 0.7f;   val m11 = 0.3f;   val m12 = 0f
        val m20 = 0f;     val m21 = 0.3f;   val m22 = 0.7f

        val colorMatrix = ColorMatrix(
            floatArrayOf(
                m00, m01, m02, 0f, 0f,
                m10, m11, m12, 0f, 0f,
                m20, m21, m22, 0f, 0f,
                0f,  0f,  0f,  1f, 0f
            )
        )

        val result = Bitmap.createBitmap(source.width, source.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        val paint = Paint()
        paint.colorFilter = ColorMatrixColorFilter(colorMatrix)
        canvas.drawBitmap(source, 0f, 0f, paint)
        return result
    }

    // Monocromatico
    private fun applyMonochrome(source: Bitmap, threshold: Int = 128): Bitmap {
        val width = source.width
        val height = source.height
        val result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        val pixels = IntArray(width * height)
        source.getPixels(pixels, 0, width, 0, 0, width, height)

        for (i in pixels.indices) {
            val c = pixels[i]
            val r = (c shr 16) and 0xFF
            val g = (c shr 8) and 0xFF
            val b = c and 0xFF
            val gray = (0.299 * r + 0.587 * g + 0.114 * b).toInt()
            val bw = if (gray >= threshold) 0xFFFFFFFF.toInt() else 0xFF000000.toInt()
            pixels[i] = bw
        }

        result.setPixels(pixels, 0, width, 0, 0, width, height)
        return result
    }

    fun saveImage(bitmap: Bitmap): String {
        val fileName = "img_${System.currentTimeMillis()}.jpg"
        val file = File(filesDir, fileName)

        val fos = FileOutputStream(file)
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos)
        fos.close()

        Log.d(TAG, "Imagen guardada en: ${file.absolutePath}")
        return file.absolutePath
    }

    private val startForResult =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->

            if (result.resultCode == Activity.RESULT_OK) {
                try {
                    Log.d(TAG, "Foto capturada exitosamente")

                    val intent = result.data
                    val photoBitmap = intent?.extras?.get("data") as? Bitmap

                    if (photoBitmap == null) {
                        Log.e(TAG, "Bitmap es nulo")
                        Toast.makeText(this, "Error: imagen no capturada", Toast.LENGTH_LONG).show()
                        return@registerForActivityResult
                    }

                    Log.d(TAG, "Bitmap obtenido: ${photoBitmap.width}x${photoBitmap.height}")

                    val detectedBitmap = runInference(photoBitmap)

                    val imagePath = saveImage(detectedBitmap)

                    val imageView = findViewById<ImageView>(R.id.imageView)
                    imageView.setImageBitmap(detectedBitmap)

                    Toast.makeText(this, "Imagen procesada", Toast.LENGTH_SHORT).show()

                } catch (e: Exception) {
                    Log.e(TAG, "Error procesando resultado: ${e.message}")
                    e.printStackTrace()
                    Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_LONG).show()
                }
            } else {
                Log.d(TAG, "Captura cancelada")
            }
        }

    override fun onDestroy() {
        super.onDestroy()
        if (::interpreter.isInitialized) {
            interpreter.close()
        }
    }
}

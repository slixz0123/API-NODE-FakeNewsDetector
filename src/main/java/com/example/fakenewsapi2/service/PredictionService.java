package com.example.fakenewsapi2.service;

import com.example.fakenewsapi2.analysis.ContextualAnalysis;
import com.example.fakenewsapi2.analysis.ContextualAnalyzer;
import com.example.fakenewsapi2.model.Prediction;
import com.example.fakenewsapi2.repository.PredictionRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Date;
import java.util.Map;

@Service
public class PredictionService {
    private final WebClient client;
    private final String pythonPath;
    private static final Logger logger = LoggerFactory.getLogger(PredictionService.class);
    private final PredictionRepository predictionRepository;

    public PredictionService(
            WebClient.Builder builder,
            @Value("${model.service.url}") String tfUrl,
            @Value("${python.scripts.path}") String pythonPath,
            PredictionRepository predictionRepository
    ) {
        this.client = builder.baseUrl(tfUrl).build();
        this.pythonPath = pythonPath;
        this.predictionRepository = predictionRepository;
    }

    public Prediction predictTF(Prediction p) {
        try {
            @SuppressWarnings("unchecked")
            Map<String, Object> resp = client.post()
                    .uri("/predict_tf")
                    .bodyValue(Map.of("title", p.getTitle(), "text", p.getText()))
                    .retrieve()
                    .bodyToMono(Map.class)
                    .block();

            boolean isFake = (Boolean) resp.get("isFake");
            double confidence = ((Number) resp.get("confidence")).doubleValue();

            p.setFake(isFake);
            p.setConfidence(confidence);

            logger.info("TF prediction - Title: {}, Fake: {}, Confidence: {}",
                    p.getTitle(), isFake, confidence);
            return p;
        } catch (Exception ex) {
            logger.error("TF prediction failed: {}", ex.getMessage());
            throw new RuntimeException("TensorFlow service error", ex);
        }
    }

    public Prediction predictLR(Prediction p) throws IOException, InterruptedException {
        return runPythonScript(p, "fake_news_predictor.py");
    }

    public Prediction predictXGB(Prediction p) throws IOException, InterruptedException {
        return runPythonScript(p, "fake_news_predictor_v2.py");
    }

    private Prediction runPythonScript(Prediction p, String scriptName)
            throws IOException, InterruptedException {
        logger.info("Executing Python script: {}", scriptName);
        ProcessBuilder pb = new ProcessBuilder(
                "python",
                pythonPath + "/" + scriptName,
                p.getTitle(),
                p.getText()
        );

        pb.redirectErrorStream(true);
        Process process = pb.start();

        InputStream inputStream = process.getInputStream();
        BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
        String output = reader.readLine();

        int exitCode = process.waitFor();
        if (exitCode != 0) {
            StringBuilder errorOutput = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                errorOutput.append(line).append("\n");
            }
            throw new RuntimeException("Script execution failed with code: " + exitCode +
                    "\nError: " + errorOutput);
        }

        return parsePythonOutput(p, output);
    }

    private Prediction parsePythonOutput(Prediction p, String output) {
        try {
            String[] parts = output.trim().split(" ");
            int label = Integer.parseInt(parts[0]);
            double confidence = Double.parseDouble(parts[1]);

            p.setFake(label == 1);
            p.setConfidence(confidence);
            return p;
        } catch (Exception ex) {
            throw new RuntimeException("Error parsing Python output: " + output, ex);
        }
    }

    public String retrainTF() {
        try {
            return client.post()
                    .uri("/retrain_tf")
                    .retrieve()
                    .bodyToMono(String.class)
                    .block();
        } catch (Exception ex) {
            throw new RuntimeException("Error calling retrain endpoint: " + ex.getMessage(), ex);
        }
    }

    public Prediction ensemblePredict(Prediction incoming) {
        Prediction tf = predictTF(incoming);
        Prediction lr;
        Prediction xgb;

        try {
            lr = predictLR(incoming);
            xgb = predictXGB(incoming);
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException("Error en predicción con modelos LR/XGB", e);
        }

        double confidence = (tf.getConfidence() * 0.5) +
                (lr.getConfidence() * 0.3) +
                (xgb.getConfidence() * 0.2);

        incoming.setFake(confidence > 0.6);
        incoming.setConfidence(confidence);
        incoming.setDate(new Date());

        return predictionRepository.save(incoming);
    }

    // === NUEVO: Predicción con análisis contextual ===
    public Prediction enhancedPredict(Prediction incoming) {
        Prediction prediction = predictTF(incoming);

        ContextualAnalyzer analyzer = new ContextualAnalyzer();
        ContextualAnalysis analysis = analyzer.analyze(incoming.getText());

        double adjustedConfidence = adjustConfidence(
                prediction.getConfidence(),
                analysis
        );

        prediction.setConfidence(adjustedConfidence);
        return predictionRepository.save(prediction);
    }

    private double adjustConfidence(double confidence, ContextualAnalysis analysis) {
        if (analysis.containsRedFlags()) {
            return confidence * 0.85;
        }
        if (analysis.passesVerificationChecks()) {
            return Math.min(1.0, confidence * 1.15);
        }
        return confidence;
    }
}

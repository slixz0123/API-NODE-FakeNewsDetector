package com.example.fakenewsapi2.analysis;




public class ContextualAnalyzer {
    public ContextualAnalysis analyze(String text) {
        ContextualAnalysis analysis = new ContextualAnalysis();

        // Bandera roja: uso excesivo de MAYÚSCULAS
        if (text.matches(".*[A-Z]{5,}.*")) {
            analysis.setContainsRedFlags(true);
        }

        // Verificación: si contiene nombres de medios confiables
        if (text.matches(".*(bbc|cnn|reuters|el\\s?país|the\\s?guardian).*")) {
            analysis.setPassesVerificationChecks(true);
        }

        return analysis;
    }
}
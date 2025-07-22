package com.example.fakenewsapi2.analysis;


public class ContextualAnalysis {
    private boolean containsRedFlags = false;
    private boolean passesVerificationChecks = false;

    public boolean containsRedFlags() {
        return containsRedFlags;
    }

    public void setContainsRedFlags(boolean value) {
        this.containsRedFlags = value;
    }

    public boolean passesVerificationChecks() {
        return passesVerificationChecks;
    }

    public void setPassesVerificationChecks(boolean value) {
        this.passesVerificationChecks = value;
    }
}
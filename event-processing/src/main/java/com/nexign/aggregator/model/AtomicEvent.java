package com.nexign.aggregator.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class AtomicEvent {

    public AtomicEvent() {
    }

    public AtomicEvent(String id, String basePhoto, String cameraPhoto, String datetime, String similarity) {
        this.id = id;
        this.basePhoto = basePhoto;
        this.cameraPhoto = cameraPhoto;
        this.dateTime = datetime;
        this.similarity = similarity;
    }

    @JsonProperty("id")
    private String id;

    @JsonProperty("basePhoto")
    private String basePhoto;

    @JsonProperty("cameraPhoto")
    private String cameraPhoto;

    @JsonProperty("dateTime")
    private String dateTime;

    @JsonProperty("similarity")
    private String similarity;
}

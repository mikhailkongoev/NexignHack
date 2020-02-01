package com.nexign.aggregator.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class BusinessEvent {

    public BusinessEvent() {
    }

    public BusinessEvent(String id, String basePhoto, String cameraPhoto, String dateTime, String similarity) {
        this.id = id;
        this.basePhoto = basePhoto;
        this.cameraPhoto = cameraPhoto;
        this.dateTime = dateTime;
        this.similarity = similarity;
    }

    private String id;

    @JsonProperty("basePhoto")
    private String basePhoto;

    @JsonProperty("cameraPhoto")
    private String cameraPhoto;

    @JsonProperty("dateTime")
    private String dateTime;

    private String similarity;
}

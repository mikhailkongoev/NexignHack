package com.nexign.wsinteractionservice.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class OutputMessage {
    private String id;
    private String basePhoto;
    private String cameraPhoto;
    private String dateTime;
    private float similarity;
}

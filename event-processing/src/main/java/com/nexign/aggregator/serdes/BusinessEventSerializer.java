package com.nexign.aggregator.serdes;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.apache.kafka.common.serialization.Serializer;

import java.util.Map;

@Slf4j
public class BusinessEventSerializer implements Serializer {

    private final ObjectMapper mapper = new ObjectMapper();

    @Override
    public void configure(Map configs, boolean isKey) {
        // no things to configure
    }

    @Override
    public byte[] serialize(String topic, Object data) {
        try {
            String val = mapper.writeValueAsString(data);
            return val.getBytes();
        } catch (JsonProcessingException e) {
            log.info("Error while serializing data {} with topic {}", data, topic);
        }
        return new byte[0];
    }

    @Override
    public void close() {
        // do nothing useful
    }
}

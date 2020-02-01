### Builld image for deploying through TeamCity pipeline
## simple command to run:  docker build -t vatemplate . && docker run -p 8080:8080 vatemplate
# build
FROM maven:3.6.2-jdk-8 as builder
RUN mkdir -p /dbuild
WORKDIR /dbuild
COPY pom.xml /dbuild
#RUN mvn -B dependency:resolve dependency:resolve-plugins
RUN mvn -B -e -C -T 1C org.apache.maven.plugins:maven-dependency-plugin:3.0.2:go-offline
COPY src /dbuild/src
RUN mvn package -DskipTests

#run
FROM openjdk:8-slim as runtime
EXPOSE 8080
ENV APP_HOME /app
ENV JAVA_OPTS=""
#ENV PG_HOST=""
RUN mkdir $APP_HOME
RUN mkdir $APP_HOME/config
RUN mkdir $APP_HOME/log
VOLUME $APP_HOME/log
VOLUME $APP_HOME/config
WORKDIR $APP_HOME
COPY --from=builder /dbuild/target/*.jar app.jar
ENTRYPOINT [ "sh", "-c", "java $JAVA_OPTS -Dserver.port=8080 -Djava.net.preferIPv4Stack=true  -jar app.jar" ]